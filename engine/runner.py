from models.settings import set
from engine.trainer import train, load_trained_model, prepare_centroids
from engine.tester import test

def run(args):
    set(args)

    # train or load trained model
    if args["train"]:
        train(args)
    else:
        load_trained_model(args)

    # compute or check if the model has two centroids
    if args["update_centroids"]:
        prepare_centroids(args)
    if (args["trained_model"].normal_img_normal_text_centroid is None or
            args["trained_model"].normal_img_abnormal_text_centroid is None):
        args["logger"].error("Centroids are not computed. Cannot calculate anomaly scores.")

    # test
    if args["test"]:
        test(args)