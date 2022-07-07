from external.mavl.inference.save_predictions import SavePKLFormat
import pickle
import os


class SaveProposalBoxes(SavePKLFormat):
    def __init__(self):
        super(SaveProposalBoxes, self).__init__()

    def save(self, save_path):
        for i, image_name in enumerate(self.predictions.keys()):
            with open(f"{save_path}/{image_name.split('.')[0]}.pkl", "wb") as f:
                img_to_boxes = self.predictions[image_name]
                pickle.dump(img_to_boxes, f)

    def save_imagenet(self, save_path):
        for i, image_name in enumerate(self.predictions.keys()):
            image_id = image_name.split('.')[0]
            class_id = image_id.split('_')[0]
            if not os.path.exists(f"{save_path}/{class_id}"):
                os.makedirs(f"{save_path}/{class_id}")
            with open(f"{save_path}/{class_id}/{image_id}.pkl", "wb") as f:
                img_to_boxes = self.predictions[image_name]
                pickle.dump(img_to_boxes, f)
