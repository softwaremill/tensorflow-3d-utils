import tensorflow as tf


def crop_and_resize(
        image,
        boxes,
        box_indices,
        crop_size
):
    image_shape = tf.shape(image)
    batch_size = image_shape[0]
    image_depth = tf.cast(image_shape[1], tf.int32)
    image_height = tf.cast(image_shape[2], tf.int32)
    image_width = tf.cast(image_shape[3], tf.int32)
    images_by_indices = tf.gather(image, box_indices)

    # In normalized coordinates (z2, y2, x2) are inside the box, but in pixel coordinates they
    # should be inside the box
    shift = tf.constant([0, 0, 0, 1, 1, 1])

    boxes_unnormalized = tf.cast(
        tf.math.multiply(
            boxes,
            [
                image_depth - 1,
                image_height - 1,
                image_width - 1,
                image_depth - 1,
                image_height - 1,
                image_width - 1
            ]
        ),
        tf.int32
    ) + shift

    def crop_step(i, result):
        to_concat = images_by_indices[i][boxes_unnormalized[i][0]:boxes_unnormalized[i][3],
                    boxes_unnormalized[i][1]:boxes_unnormalized[i][4],
                    boxes_unnormalized[i][2]:boxes_unnormalized[i][5], :]
        resized_xy = tf.image.resize_bilinear(to_concat, crop_size[1:])
        transposed = tf.transpose(resized_xy, (1, 0, 2, 3))
        resized_xyz = tf.image.resize_bilinear(transposed, crop_size[:2])
        original_shape_resized = tf.transpose(resized_xyz, (1, 0, 2, 3))
        return [tf.add(i, 1), tf.concat([result, [original_shape_resized]], 0)]

    result = tf.zeros([0, crop_size[0], crop_size[1], crop_size[2], 1], tf.float32)

    i = tf.constant(0)

    [i, result] = tf.while_loop(
        lambda i, _: i < batch_size,
        lambda i, result: crop_step(i, result),
        [i, result],
        shape_invariants=[
            i.get_shape(),
            tf.TensorShape([None, crop_size[0], crop_size[1], crop_size[2], None])
        ],
        name="crop_and_resize_while"
    )

    return result[1:]


def non_max_suppression(boxes,
                        scores,
                        max_output_size,
                        iou_threshold=0.5,
                        score_threshold=float('-inf')):
    number_of_processed = tf.constant(0)
    boxes_size = tf.shape(boxes)[0]

    indices = tf.zeros([0], tf.int32)

    def iou(a, b):
        zA = tf.maximum(a[0], b[0])
        yA = tf.maximum(a[1], b[1])
        xA = tf.maximum(a[2], b[2])
        zB = tf.minimum(a[3], b[3])
        yB = tf.minimum(a[4], b[4])
        xB = tf.minimum(a[5], b[5])

        interArea = tf.maximum(0.0, xB - xA) * tf.maximum(0.0, yB - yA) * tf.maximum(0.0, zB - zA)

        boxAArea = (a[3] - a[0]) * (a[4] - a[1]) * (a[5] - a[2])
        boxBArea = (b[3] - b[0]) * (b[4] - b[1]) * (b[5] - b[2])

        return interArea / (boxAArea + boxBArea - interArea)

    def iou_evaluation(n, boxes, scores, highest_score_box, index, removed_boxes):
        iou_value = iou(highest_score_box, boxes[index])

        (boxes, scores, index, removed_boxes) = tf.cond(
            tf.greater(iou_value, iou_threshold),
            lambda: (
                tf.concat([boxes[:index], boxes[index + 1:]], 0),
                tf.concat([scores[:index], scores[index + 1:]], 0),
                index,
                tf.add(removed_boxes, 1)
            ),
            lambda: (boxes, scores, tf.add(index, 1), removed_boxes)
        )
        return [tf.add(n, 1), boxes, scores, highest_score_box, index, removed_boxes]

    def nms_step(n, boxes, scores, proposals):
        sorted_args = tf.argsort(scores, direction='DESCENDING')
        highest_score_arg = sorted_args[0]
        highest_score_box = boxes[highest_score_arg]
        number_of_processed = tf.constant(0)
        index = tf.constant(0)
        boxes = tf.concat([boxes[:highest_score_arg], boxes[highest_score_arg + 1:]], 0)
        scores = tf.concat([scores[:highest_score_arg], scores[highest_score_arg + 1:]], 0)
        boxes_size = tf.shape(boxes)[0]
        removed_boxes = tf.constant(0)

        (n_internal, boxes, scores, highest_score_box, index, removed_boxes) = tf.while_loop(
            lambda n, b, s, h, i, r: n < boxes_size,
            lambda n, b, s, h, i, r: iou_evaluation(n, b, s, h, i, r),
            [
                number_of_processed,
                boxes,
                scores,
                highest_score_box,
                index,
                removed_boxes
            ],
            shape_invariants=[
                number_of_processed.get_shape(),
                tf.TensorShape([None, 6]),
                tf.TensorShape([None]),
                tf.TensorShape([None]),
                index.get_shape(),
                removed_boxes.get_shape()
            ]
        )
        return [
            tf.add(tf.add(n, 1), removed_boxes),
            boxes,
            scores,
            tf.concat([proposals, [highest_score_arg]], 0)
        ]

    number_of_processed, boxes, scores, indices = tf.while_loop(
        lambda n, b, s, p: tf.math.logical_and(n < boxes_size, tf.shape(p)[0] < max_output_size),
        lambda n, b, s, p: nms_step(n, b, s, p),
        [number_of_processed, boxes, scores, indices],
        shape_invariants=[
            number_of_processed.get_shape(),
            tf.TensorShape([None, 6]),
            tf.TensorShape([None]),
            tf.TensorShape([None]),
        ]
    )

    return indices


