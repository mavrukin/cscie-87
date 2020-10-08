# cython: language_level=3
import argparse
import csv
import datetime
import matplotlib.pyplot as plt
import numpy
import os
import os.path as path
import sklearn.linear_model as linear_model
from array import array
from itertools import compress
from sklearn.feature_selection import RFE


def get_num_lines_from_csv(csv_file_path):
    line_count_no_header = 0
    max_row_keys = 0
    if path.isfile(csv_file_path) and os.access(csv_file_path, os.R_OK):
        with open(csv_file_path, "r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for line in csv_reader:
                row_keys = 0
                for key in line.keys():
                    if key != "Wait" and not str(key).startswith("x_"):
                        row_keys +=1
                if row_keys > max_row_keys:
                    max_row_keys = row_keys
                line_count_no_header += 1
    return line_count_no_header, max_row_keys


def process_csv_file(csv_file_path):
    num_samples_in_file, num_row_keys = get_num_lines_from_csv(csv_file_path)
    if path.isfile(csv_file_path) and os.access(csv_file_path, os.R_OK):
        with open(csv_file_path, "r") as csv_file:
            wait_times = array('i')
            features = dict()   # [str(Feature Name) --> Array(Values)
            combined_features_array = []
            line_counter = 0
            total_elements = 0
            csv_reader = csv.DictReader(csv_file)
            column_id_to_name = dict()
            for line in csv_reader:
                if line_counter % 1000 == 0:
                    print(f"CSV Loading Line: [{line_counter}]")
                grouped_features = numpy.empty(num_row_keys)
                key_counter = 0
                for key in line.keys():
                    if key == "Wait":
                        wait_times.append(int(line["Wait"]))
                    elif not str(key).startswith("x_"):
                        if key not in features:
                            features[key] = numpy.empty(num_samples_in_file)
                        feature_value = float(line[key]) if line[key].find(".") >= 0 else int(line[key])
                        features[key][line_counter] = feature_value
                        grouped_features[key_counter] = feature_value
                        column_id_to_name[key_counter] = key
                        key_counter += 1
                combined_features_array.append(grouped_features)
                line_counter += 1

            # combined_features_array = numpy.asarray(combined_features_array)

            print(f"Num Rows: {line_counter} num array elements: {len(combined_features_array)}")
            model = linear_model.LinearRegression()
            model.fit(combined_features_array, wait_times)
            model_prediction = model.predict(combined_features_array)
            residual = wait_times - model_prediction
            error = abs(residual).mean()
            print(f"Error: {error}")
            plt.scatter(model_prediction, wait_times, alpha=0.1)
            plt.show()

            print(f"Python feature selection:")
            for num_features in range(1, 4):
                rfe = RFE(model, n_features_to_select = num_features)
                selected_features = rfe.fit_transform(combined_features_array, wait_times)
                # Fitting data to model
                model.fit(selected_features, wait_times)
                print(f"rfe.support_ : {rfe.support_}")
                print(f"rfe.ranking_ : {rfe.ranking_}")
                columns = list(compress(sorted(column_id_to_name.keys()), rfe.support_))
                refit_features = []
                for combined_features in combined_features_array:
                    # refit_features.append([])
                    model_features = []
                    column_counter = 0
                    for column in columns:
                        model_features.insert(column_counter, combined_features[column])
                        column_counter += 1
                    refit_features.append(model_features)
                print(f"Num Features: {len(features[column_id_to_name[column]])}")
                print(f"Feature: {column_id_to_name[column]}")
                print(f"Num Wait Times: {len(wait_times)}")
                model.fit(refit_features, wait_times)
                rfe_error = abs(wait_times - model.predict(refit_features)).mean()
                print(f"RFE Error: {rfe_error}, columns: {columns},")
                for column in columns:
                    print(f"mapped to names: {column_id_to_name[column]}")

            print(f"Computing Best 3-tuple feature combination")
            three_features_min_error = float('inf')
            three_features_min_error_tuple = None
            total_loops = pow(len(column_id_to_name.keys()), 3)
            loop_counter = 0
            start_time = datetime.datetime.now()
            for f1_feature_id in column_id_to_name.keys():
                for f2_feature_id in column_id_to_name.keys():
                    if f1_feature_id != f2_feature_id:
                        for f3_feature_id in column_id_to_name.keys():
                            loop_counter += 1
                            if loop_counter % 1000 == 0:
                                print(f"Progress [{loop_counter} / {total_loops}] [{datetime.datetime.now() - start_time}]")
                            if f3_feature_id != f1_feature_id and f3_feature_id != f2_feature_id:
                                three_features_vector_transpose = numpy.transpose([features[column_id_to_name[f1_feature_id]],
                                     features[column_id_to_name[f2_feature_id]],
                                     features[column_id_to_name[f3_feature_id]]])
                                model.fit(three_features_vector_transpose, wait_times)
                                three_feature_error = abs(wait_times - model.predict(three_features_vector_transpose)).mean()
                                if three_feature_error < three_features_min_error:
                                    three_features_min_error = three_feature_error
                                    three_features_min_error_tuple = (f1_feature_id, f2_feature_id, f3_feature_id)
                                    print(f"Found New Minimum - error: {three_features_min_error}, tuple: {three_features_min_error_tuple}")

            print(f"Best 3-feature error:")
            print(f"\t3-Feature Error: {three_features_min_error}")
            print(f"\t3-Feature Error Tuple: {three_features_min_error_tuple}")
            if three_features_min_error_tuple is not None:
                print(f"3-Feature Names: {column_id_to_name[three_features_min_error_tuple[0]]}, {column_id_to_name[three_features_min_error_tuple[1]]}, {column_id_to_name[three_features_min_error_tuple[2]]}")

            selected_feature_ids = set()
            feature_vector = []
            for num_features_to_select in range(0, 15):
                feature_id_with_min_error = None
                feature_vector_min_error = float('inf')
                for feature_id in column_id_to_name.keys():
                    if feature_id not in selected_feature_ids:
                        sample_counter = 0
                        for combined_features in combined_features_array:
                            if len(feature_vector) <= sample_counter:
                                feature_vector.insert(sample_counter, [])
                            if len(feature_vector[sample_counter]) <= num_features_to_select:
                                feature_vector[sample_counter].insert(num_features_to_select, combined_features[feature_id])
                            else:
                                feature_vector[sample_counter][num_features_to_select] = combined_features[feature_id]
                            sample_counter += 1

                        model.fit(feature_vector, wait_times)
                        feature_error = abs(wait_times - model.predict(feature_vector)).mean()

                        if feature_error < feature_vector_min_error:
                            feature_vector_min_error = feature_error
                            feature_id_with_min_error = feature_id

                print(f"Num Features: {num_features_to_select + 1}")
                print(f"\tMin Error: {feature_vector_min_error}, id: {feature_id_with_min_error}, feature: {column_id_to_name[feature_id_with_min_error]}")
                selected_feature_ids.add(feature_id_with_min_error)
                sample_counter = 0
                for combined_features in combined_features_array:
                    feature_vector[sample_counter][num_features_to_select] = combined_features[feature_id_with_min_error]
                    sample_counter += 1

            print(f"Total Lines: {line_counter} with Total Cells: {total_elements}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV Data File")
    parser.add_argument('--csv_data_file', type=str, required=True,
                        help="Path to csv containing data records")
    args = parser.parse_args()
    process_csv_file(args.csv_data_file)