import numpy as np
from multiprocessing import Process, Manager


# bbox = target['bbox']
#                 bbox = torch.stack([bbox[:, :, 1], bbox[:, :, 0], bbox[:, :, 3], bbox[:, :, 2]],
#                                    dim=-1)  # Change yxyx to xyxy
#                 cls = torch.unsqueeze(target['cls'], dim=-1)
#                 max_cls = torch.sum(cls > 0, dim=1).max().item()
#                 target_tensor = torch.cat([bbox, cls], dim=-1)[:, :max_cls, :]
#                 output = output['detections']
#                 output = output.cpu().detach().numpy()
#                 output[:, :, :4] = output[:, :, :4] / target['img_scale'].reshape(-1, 1, 1).cpu().numpy()  # Normalized
#                 output[:, :, 2] = output[:, :, 2] + output[:, :, 0]  # Change xywh to xyxy
#                 output[:, :, 3] = output[:, :, 3] + output[:, :, 1]
#                 target_tensor = target_tensor.cpu().detach().numpy()
#
#                 evaluator.add_predictions(output, target_tensor)

class MeanAveragePrecision(object):
    def __init__(self, n_class, iou_array, score_th=0.1, multiprocessing=False):
        self.n_class = n_class + 1
        self.n_iou = len(iou_array)
        self.iou_array = iou_array
        self.zero = np.zeros([1, self.n_iou]).reshape(1, self.n_iou)
        self.ones = np.ones([1, self.n_iou]).reshape(1, self.n_iou)
        self.multiprocessing = multiprocessing
        self.process = []
        self.score_th = score_th
        self.eps = 1e-6
        if not self.multiprocessing:
            self.data_per_class = [[] for i in range(self.n_class)]
            self.n_samples_per_class = np.zeros(self.n_class)
        else:
            self.manger = Manager()
            self.data_per_class = self.manger.list()
            self.n_samples_per_class = self.manger.list()
            [self.data_per_class.append([]) for i in range(self.n_class)]

    @staticmethod
    def area(box_a: np.ndarray):
        return (box_a[:, :, 2] - box_a[:, :, 0]) * (box_a[:, :, 3] - box_a[:, :, 1])

    def compute_iou(self,box_a: np.ndarray, box_b: np.ndarray):
        """

        :param box_a: [B,P,4]
        :param box_b: [B,L,5]
        :return:
        """
        area_a = np.expand_dims(MeanAveragePrecision.area(box_a), axis=1)
        area_b = np.expand_dims(MeanAveragePrecision.area(box_b), axis=2)
        box_a = np.expand_dims(box_a, axis=1)
        box_b = np.expand_dims(box_b, axis=2)
        xy_max = np.maximum(box_a[:, :, :, :2], box_b[:, :, :, :2])
        xy_min = np.minimum(box_a[:, :, :, 2:], box_b[:, :, :, 2:])
        inter_area = np.maximum(0, xy_min[:, :, :, 0] - xy_max[:, :, :, 0]) * np.maximum(0,
                                                                                         xy_min[:, :, :, 1] - xy_max[:,
                                                                                                              :, :, 1])
        iou = inter_area / ((area_a + area_b - inter_area) + self.eps)
        return iou

    def _calculate_sample_tp(self, prediction_i, label_i, iou_i):
        for class_index in range(self.n_class):  # Loop over classes
            prediction_class_status = prediction_i[:, 5] == class_index
            if np.any(prediction_class_status):
                res_list = []
                prediction_index = np.where(prediction_class_status)[0]
                label_class_status = label_i[:, 4] == class_index
                if np.any(label_class_status):
                    label_index = np.where(label_class_status)[0]

                    label_status = np.ones([label_index.shape[0], self.n_iou], dtype=np.bool)
                    for pi in prediction_index:
                        iou = iou_i[label_index, pi]
                        iou_index = np.argmax(iou)
                        max_iou = iou[iou_index]
                        score = prediction_i[pi, 4]
                        if score > self.score_th:
                            tp = (self.iou_array < max_iou) * label_status[iou_index, :]
                            label_status[iou_index, :] = label_status[iou_index, :] * np.logical_not(tp)
                            res_list.append(np.concatenate([tp, np.asarray(score).reshape(1)]))
                else:
                    for pi in prediction_index:
                        score = prediction_i[pi, 4]
                        res_list.append(
                            np.concatenate([np.zeros([self.n_iou], dtype=np.bool), np.asarray(score).reshape(1)]))
                self.data_per_class[class_index].extend(res_list)
        if self.multiprocessing:
            self.n_samples_per_class.append(np.bincount(label_i[:, -1].astype('int32'), minlength=self.n_class))
        else:
            self.n_samples_per_class += np.bincount(label_i[:, -1].astype('int32'), minlength=self.n_class)

    def add_predictions(self, prediction: np.ndarray, label: np.ndarray):
        """

        :param prediction: A tensor of shape [B,P,6] where 6 is box,score,class
        :param label: A tensor of shape [B,L,5] where 5 is box,class
        :return:
        """
        iou_matrix = self.compute_iou(prediction[:, :, :4], label[:, :, :4])
        for i in range(prediction.shape[0]):  # Loop Over samples
            prediction_i = prediction[i, :]
            label_i = label[i, :]
            iou_i = iou_matrix[i, :]
            if self.multiprocessing:
                p = Process(target=self._calculate_sample_tp, args=(prediction_i, label_i, iou_i))  # Passing the list
                p.start()
            else:
                self._calculate_sample_tp(prediction_i, label_i, iou_i)

    def evaluate(self):
        per_class_list = []
        per_class_scale = []
        for class_index, class_data in enumerate(self.data_per_class):
            if len(class_data) == 0:
                ap = np.nan
            else:
                class_data = np.stack(class_data, axis=0)
                sort_index = np.argsort(class_data[:, -1])[::-1]
                tp_array = class_data[sort_index, :-1]

                n_samples = self.n_samples_per_class[class_index]
                per_class_scale.append(n_samples)
                cum_true_positives = np.cumsum(tp_array, axis=0)
                cum_false_positives = np.cumsum(1 - tp_array, axis=0)
                precision = cum_true_positives.astype(float) / (
                        cum_true_positives + cum_false_positives+ self.eps)
                recall = cum_true_positives.astype(float) / (n_samples+ self.eps)

                recall = np.concatenate(
                    [self.zero, recall, self.ones])
                precision = np.concatenate([self.zero, precision, self.zero])

                # Preprocess precision to be a non-decreasing array
                for i in range(precision.shape[0] - 2, -1, -1):
                    precision[i, :] = np.maximum(precision[i, :], precision[i + 1, :])
                average_precision_list = []
                for iou_index in range(self.n_iou):
                    indices = np.where(recall[1:, iou_index] != recall[:-1, iou_index])[0] + 1
                    average_precision = np.sum(
                        (recall[indices, iou_index] - recall[indices - 1, iou_index]) * precision[indices, iou_index])
                    average_precision_list.append(average_precision)
                per_class_list.append(average_precision_list)
        return (np.mean(per_class_list), np.mean(per_class_list, axis=0))
