from sklearn.metrics import normalized_mutual_info_score
import json

def main():

    ground_truth_files = ["../resource/asnlib/publicdata/cluster1.json", \
        "../resource/asnlib/publicdata/cluster2.json", \
        "../resource/asnlib/publicdata/cluster3.json", \
        "../resource/asnlib/publicdata/cluster4.json", \
        "../resource/asnlib/publicdata/cluster5.json"]
    
    prediction_files = ["./out_file1.txt", "./out_file2.txt", "./out_file3.txt", "./out_file4.txt", "./out_file5.txt"]

    for i in range(5):
        ground_truth_file = ground_truth_files[i]
        prediction_file = prediction_files[i]
        with open(ground_truth_file, "r") as f:
            true_label_dict = json.load(f)

        with open(prediction_file, "r") as f:
            predict_label_dict = json.load(f)

        true_labels = [-1] * len(true_label_dict)
        for point_id, cluster_id in true_label_dict.items():
            true_labels[int(point_id)] = cluster_id

        predict_labels = [-1] * len(true_label_dict)
        for point_id, cluster_id in predict_label_dict.items():
            predict_labels[int(point_id)] = cluster_id

        NMI = normalized_mutual_info_score(true_labels, predict_labels)
        print("NMI: %.5f" % NMI)
        print("------------")

if __name__ == "__main__":
    main()