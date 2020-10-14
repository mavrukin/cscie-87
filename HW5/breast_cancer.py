import argparse
import csv
import numpy
import os
import os.path as path
from matplotlib import pyplot
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


def get_num_lines_from_csv(csv_file_path):
    line_count_no_header = 0
    max_row_keys = 0
    if path.isfile(csv_file_path) and os.access(csv_file_path, os.R_OK):
        with open(csv_file_path, "r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for line in csv_reader:
                row_keys = 0
                for key in line.keys():
                    if key != "id" and key != "diagnosis":
                        row_keys +=1
                if row_keys > max_row_keys:
                    max_row_keys = row_keys
                line_count_no_header += 1
    return line_count_no_header, max_row_keys


def process_breast_cancer_data(csv_data_file):
    num_samples, num_features = get_num_lines_from_csv(csv_data_file)
    samples = []
    features_to_samples = dict()
    column_index_to_name = dict()
    diagnosis = numpy.empty(num_samples, numpy.bool)
    if path.isfile(csv_data_file) and os.access(csv_data_file, os.R_OK):
        with open(csv_data_file, "r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            sample_counter = 0
            for line in csv_reader:
                features = numpy.empty(num_features)
                feature_key_counter = 0
                for key in line.keys():
                    if key == "diagnosis":
                        diagnosis[sample_counter] = True if line[key] == "M" else False
                    elif key != "id":
                        if feature_key_counter not in column_index_to_name:
                            column_index_to_name[feature_key_counter] = str(key)
                        if key not in features_to_samples:
                            features_to_samples[key] = numpy.empty(num_samples)
                        feature_value = float(line[key]) if "." in line[key] else int(line[key])
                        features[feature_key_counter] = feature_value
                        features_to_samples[key][sample_counter] = feature_value
                        feature_key_counter += 1
                samples.append(features)
                sample_counter += 1


    # Problem 5 - LogisticRegression - Best of 3
    print(f"Computing Best 3-tuple feature combination")
    three_feature_max_accuracy = 0
    three_features_min_error_tuple = None
    three_feature_max_recall = 0
    three_feature_max_recall_tuple = None
    total_loops = pow(len(column_index_to_name.keys()), 3)
    loop_counter = 0
    model = LogisticRegression()
    for f1_feature_id in column_index_to_name.keys():
        for f2_feature_id in column_index_to_name.keys():
            if f1_feature_id != f2_feature_id:
                for f3_feature_id in column_index_to_name.keys():
                    loop_counter += 1
                    if loop_counter % 1000 == 0:
                        print(f"Progress [{loop_counter} / {total_loops}]")
                    if f3_feature_id != f1_feature_id and f3_feature_id != f2_feature_id:
                        three_feature_vector_transpose = numpy.transpose([
                            features_to_samples[column_index_to_name[f1_feature_id]],
                            features_to_samples[column_index_to_name[f2_feature_id]],
                            features_to_samples[column_index_to_name[f3_feature_id]]
                        ])
                        model.fit(three_feature_vector_transpose, diagnosis)
                        model_prediction = model.predict(three_feature_vector_transpose)
                        model_accuracy = metrics.accuracy_score(diagnosis, model_prediction)
                        model_recall = metrics.recall_score(diagnosis, model_prediction)
                        if model_accuracy > three_feature_max_accuracy:
                            three_feature_max_accuracy = model_accuracy
                            three_features_min_error_tuple = (f1_feature_id, f2_feature_id, f3_feature_id)
                            print(f"Found New Max Accuracy - Accuracy: {three_feature_max_accuracy}, tuple: {three_features_min_error_tuple}")

                        if model_recall > three_feature_max_recall:
                            three_feature_max_recall = model_recall
                            three_feature_max_recall_tuple = (f1_feature_id, f2_feature_id, f3_feature_id)
                            print(f"Found New Max Recall - Recall: {three_feature_max_recall}, tuple: {three_feature_max_recall_tuple}")

    print(f"Best 3-feature Accuracy:")
    print(f"\t3-Feature Accuracy: {three_feature_max_accuracy}")
    print(f"\t3-Feature Accuracy Tuple: {three_features_min_error_tuple}")
    if three_features_min_error_tuple is not None:
        print(f"3-Feature Names: {column_index_to_name[three_features_min_error_tuple[0]]}, {column_index_to_name[three_features_min_error_tuple[1]]}, {column_index_to_name[three_features_min_error_tuple[2]]}")

    # Problem 6 - LogisticRegression - Best of 3 - Recall Calculation
    print(f"Best 3-Feature Recall:")
    print(f"\t3-Feature Recall: {three_feature_max_recall}")
    print(f"\t3-Feature Recall Tuple: {three_feature_max_recall_tuple}")
    if three_feature_max_recall_tuple is not None:
        print(f"3-Feature Names: {column_index_to_name[three_feature_max_recall_tuple[0]]}, {column_index_to_name[three_feature_max_recall_tuple[1]]}, {column_index_to_name[three_feature_max_recall_tuple[2]]}")

    # Problem 7 - DecisionTreeClassifier - Picking best 3 using a decision tree
    tree_classifier = DecisionTreeClassifier(max_leaf_nodes=3)
    tree_classifier.fit(samples, diagnosis)
    tree_prediction = tree_classifier.predict(samples)
    tree_accuracy = metrics.accuracy_score(diagnosis, tree_prediction)
    print(f"Tree Accuracy Score: {tree_accuracy}")
    pyplot.figure(figsize=(15, 10))
    feature_names_ordered = []
    for feature_id in sorted(column_index_to_name.keys()):
        feature_names_ordered.append(column_index_to_name[feature_id])
    tree.plot_tree(tree_classifier, feature_names=list(feature_names_ordered), class_names= ["B", "M"], rounded=True, fontsize=12, filled=True)
    pyplot.show()

    # Problem 8 - RandomForestClassifier
    forest_classifier = RandomForestClassifier(n_estimators=10, max_depth=3)
    forest_classifier.fit(samples, diagnosis)
    forest_prediction = forest_classifier.predict(samples)
    forest_accuracy = metrics.accuracy_score(diagnosis, forest_prediction)
    print(f"Forest Accuracy: {forest_accuracy}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV Data File")
    parser.add_argument('--csv_data_file', type=str, required=True,
                        help="Path to csv containing data records")
    args = parser.parse_args()
    process_breast_cancer_data(args.csv_data_file)