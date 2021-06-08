VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

COLORS = {'aeroplane': (0, 0, 0),
          'bicycle': (128, 0, 0),
          'bird': (0, 128, 0),
          'boat': (128, 128, 0),
          'bottle': (0, 0, 128),
          'bus': (128, 0, 128),
          'car': (0, 128, 128),
          'cat': (128, 128, 128),
          'chair': (64, 0, 0),
          'cow': (192, 0, 0),
          'diningtable': (64, 128, 0),
          'dog': (192, 128, 0),
          'horse': (64, 0, 128),
          'motorbike': (192, 0, 128),
          'person': (64, 128, 128),
          'pottedplant': (192, 128, 128),
          'sheep': (0, 64, 0),
          'sofa': (128, 64, 0),
          'train': (0, 192, 0),
          'tvmonitor': (128, 192, 0)}


class Evaluation:
    def __init__(self, predictions, targets, threshold):
        super(Evaluation, self).__init__()
        self.predictions = predictions
        self.targets = targets
        self.threshold = threshold

    @staticmethod
    def compute_ap(recall, precision):
        # average precision calculation
        recall = np.concatenate(([0.], recall, [1.]))
        precision = np.concatenate(([0.], precision, [0.]))

        for i in range(precision.size - 1, 0, -1):
            precision[i - 1] = max(precision[i - 1], precision[i])

        ap = 0.0  # average precision (AUC of the precision-recall curve).
        for i in range(precision.size - 1):
            ap += (recall[i + 1] - recall[i]) * precision[i + 1]

        return ap

    def evaluate(self):
        aps = []
        print('CLASS'.ljust(25, ' '), 'AP')
        for class_name in VOC_CLASSES:
            class_preds = self.predictions[class_name]  # [[image_id,confidence,x1,y1,x2,y2],...]
            if len(class_preds) == 0:
                ap = -1
                print('---class {} ap {}---'.format(class_name, ap))
                aps.append(ap)
                break
            # print(pred)
            image_ids = [x[0] for x in class_preds]
            confidence = np.array([float(x[1]) for x in class_preds])
            BB = np.array([x[2:] for x in class_preds])
            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            npos = 0.
            for (key1, key2) in self.targets:
                if key2 == class_name:
                    npos += len(self.targets[(key1, key2)])
            nd = len(image_ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)

            for d, image_id in enumerate(image_ids):
                bb = BB[d]
                if (image_id, class_name) in self.targets:
                    BBGT = self.targets[(image_id, class_name)]
                    for x1y1_x2y2 in BBGT:
                        # compute overlaps
                        # intersection
                        x_min = np.maximum(x1y1_x2y2[0], bb[0])
                        y_min = np.maximum(x1y1_x2y2[1], bb[1])
                        x_max = np.minimum(x1y1_x2y2[2], bb[2])
                        y_max = np.minimum(x1y1_x2y2[3], bb[3])
                        w = np.maximum(x_max - x_min + 1., 0.)
                        h = np.maximum(y_max - y_min + 1., 0.)
                        intersection = w * h

                        union = (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + (x1y1_x2y2[2] - x1y1_x2y2[0] + 1.) * (
                                x1y1_x2y2[3] - x1y1_x2y2[1] + 1.) - intersection
                        if union == 0:
                            print(bb, x1y1_x2y2)

                        overlaps = intersection / union
                        if overlaps > self.threshold:
                            tp[d] = 1
                            BBGT.remove(x1y1_x2y2)
                            if len(BBGT) == 0:
                                del self.targets[(image_id, class_name)]
                            break
                    fp[d] = 1 - tp[d]
                else:
                    fp[d] = 1
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            recall = tp / float(npos)
            precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

            ap = self.compute_ap(recall, precision)

            print(f'{class_name}'.ljust(25, ' '), f'{ap:.2f}')
            aps.append(ap)

        return aps


if __name__ == '__main__':
    from utils.predict import *
    from collections import defaultdict
    from tqdm import tqdm

    targets = defaultdict(list)
    predictions = defaultdict(list)
    image_list = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    im_show = False

    print('DATA PREPARING...')
    with open('../Dataset/test.txt') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        image_name = f'{line}.jpg'
        image_list.append(image_name)

        with open(f'../Dataset/Labels/{line}.txt') as f:
            objects = f.readlines()

        for object in objects:
            c, x1, y1, x2, y2 = map(int, object.rstrip().split())
            class_name = VOC_CLASSES[c]
            targets[(image_name, class_name)].append([x1, y1, x2, y2])
    print('DONE.\n')
    print('START TESTING...')

    model = resnet50().to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load('best.pth')['state_dict'])
    model.eval()

    # image_list = image_list[:500]
    for image_name in tqdm(image_list):

        result = predict_gpu(model, image_name, root_path='../Dataset/Images/')

        for (x1, y1), (x2, y2), class_name, image_name, conf in result:
            predictions[class_name].append([image_name, conf, x1, y1, x2, y2])

        if im_show:
            image = cv2.imread('../Dataset/Images/' + image_name)

            for x1y1, x2y2, class_name, _, prob in result:
                color = COLORS[class_name]
                cv2.rectangle(image, x1y1, x2y2, color, 2)

                label = class_name + str(round(prob, 2))
                text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

                p1 = (x1y1[0], x1y1[1] - text_size[1])
                cv2.rectangle(image, (p1[0] - 2 // 2, p1[1] - 2 - baseline),
                              (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)

                cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                            8)

            cv2.imshow('Prediction', image)
            cv2.waitKey(0)

    if not im_show:
        print('\nSTART EVALUATION...')

        aps = Evaluation(predictions, targets, threshold=0.4).evaluate()
        print(f'mAP: {np.mean(aps):.2f}')
        print('\nDONE.')
