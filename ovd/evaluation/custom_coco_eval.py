import itertools
import numpy as np
from tabulate import tabulate

from detectron2.evaluation.coco_evaluation import COCOEvaluator
from detectron2.utils.logger import create_small_table
from ..datasets.coco_zeroshot import categories_seen, categories_unseen
from .coco_ovd_frequencly_splits import category_instance_frequency


def get_frequency_distr(class_ids):
    frequent_thresh = 10000
    common_thresh = 2000
    frequent_categores = []
    common_categories = []
    rare_categories = []
    for name in class_ids:
        frequency = category_instance_frequency[name]
        if frequency > frequent_thresh: frequent_categores.append(name)
        if common_thresh < frequency <= frequent_thresh: common_categories.append(name)
        if frequency <= common_thresh: rare_categories.append(name)
    return frequent_categores, common_categories, rare_categories


class CustomCOCOEvaluator(COCOEvaluator):
    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Additionally plot mAP for 'seen classes' and 'unseen classes'
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        seen_names = set([x['name'] for x in categories_seen])
        unseen_names = set([x['name'] for x in categories_unseen])
        results_per_category = []
        results_per_category50 = []
        results_per_category50_seen = []
        results_per_category50_unseen = []
        frequency_eval = True
        if frequency_eval:
            results_seen_common = []
            results_seen_frequent = []
            results_seen_rare = []
            results_unseen_common = []
            results_unseen_frequent = []
            results_unseen_rare = []
            seen_frequent_categories, seen_common_categories, seen_rare_categories = get_frequency_distr(seen_names)
            assert (len(seen_frequent_categories) + len(seen_common_categories) + len(seen_rare_categories)) == 48
            unseen_frequent_categories, unseen_common_categories, unseen_rare_categories = get_frequency_distr(
                unseen_names)
            assert (len(unseen_frequent_categories) + len(unseen_common_categories) + len(unseen_rare_categories)) == 17
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))
            precision50 = precisions[0, :, idx, 0, -1]
            precision50 = precision50[precision50 > -1]
            ap50 = np.mean(precision50) if precision50.size else float("nan")
            results_per_category50.append(("{}".format(name), float(ap50 * 100)))
            if name in seen_names:
                results_per_category50_seen.append(float(ap50 * 100))
                if frequency_eval:
                    if name in seen_frequent_categories: results_seen_frequent.append(float(ap50 * 100))
                    if name in seen_common_categories: results_seen_common.append(float(ap50 * 100))
                    if name in seen_rare_categories: results_seen_rare.append(float(ap50 * 100))
            if name in unseen_names:
                results_per_category50_unseen.append(float(ap50 * 100))
                if frequency_eval:
                    if name in unseen_frequent_categories: results_unseen_frequent.append(float(ap50 * 100))
                    if name in unseen_common_categories: results_unseen_common.append(float(ap50 * 100))
                    if name in unseen_rare_categories: results_unseen_rare.append(float(ap50 * 100))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        N_COLS = min(6, len(results_per_category50) * 2)
        results_flatten = list(itertools.chain(*results_per_category50))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP50"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP50: \n".format(iou_type) + table)
        self._logger.info(
            "Seen {} AP50: {}".format(
                iou_type,
                sum(results_per_category50_seen) / len(results_per_category50_seen),
            ))
        self._logger.info(
            "Unseen {} AP50: {}".format(
                iou_type,
                sum(results_per_category50_unseen) / len(results_per_category50_unseen),
            ))
        # frequent, common, rare: seen
        seen_frequent = sum(results_seen_frequent) / len(results_seen_frequent)
        seen_common = sum(results_seen_common) / len(results_seen_common)
        seen_rare = sum(results_seen_rare) / len(results_seen_rare)
        self._logger.info("Seen: freq: {}, common {}, rare {}".format(seen_frequent, seen_common, seen_rare))

        # frequent, common, rare : unseen
        unseen_frequent = sum(results_unseen_frequent) / len(results_unseen_frequent)
        unseen_common = sum(results_unseen_common) / len(results_unseen_common)
        unseen_rare = sum(results_unseen_rare) / len(results_unseen_rare)
        self._logger.info("Unseen: freq: {}, common {}, rare {}".format(unseen_frequent, unseen_common, unseen_rare))

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        results["AP50-seen"] = sum(results_per_category50_seen) / len(results_per_category50_seen)
        results["AP50-unseen"] = sum(results_per_category50_unseen) / len(results_per_category50_unseen)
        return results
