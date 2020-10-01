# Author: Michael Avrukin
# CSCI-E87 - Fall 2020
# Week 2 Homework
# Process HL7 Records in order to find:
# 1) The youngest male patient
# 2) The number of unique female patients, with uniqueness indexed on name
# For HL7 2.2, 2.3.1, and 2.5.1 this reference is used for sequence of fields:
#   https://hl7-definition.caristix.com/v2/HL7v2.5.1/Fields/PID.5

import argparse
import os
import os.path as path
from datetime import datetime


# A class meant to represent a patient's name along with all the relevant
# components as defined in HL7 2.2, 2.3.1, and 2.5.1
class PatientName:
    def __init__(self):
        self.family_name = ""
        self.given_name = ""
        self.middle_initial_or_name = ""
        self.suffix = ""
        self.prefix = ""

    @property
    def family_name(self):
        return self.__family_name

    @family_name.setter
    def family_name(self, family_name):
        self.__family_name = family_name

    @property
    def given_name(self):
        return self.__given_name

    @given_name.setter
    def given_name(self, given_name):
        self.__given_name = given_name

    @property
    def middle_initial_or_name(self):
        return self.__middle_initial_or_name

    @middle_initial_or_name.setter
    def middle_initial_or_name(self, middle_initial_or_name):
        self.__middle_initial_or_name = middle_initial_or_name

    @property
    def suffix(self):
        return self.__suffix

    @suffix.setter
    def suffix(self, suffix):
        self.__suffix = suffix

    @property
    def prefix(self):
        return self.__prefix

    @prefix.setter
    def prefix(self, prefix):
        self.__prefix = prefix

    def __str__(self):
        return f"{self.prefix} {self.family_name} {self.given_name} " \
               f"{self.middle_initial_or_name} {self.suffix}"


# A class representing a patient record w/ name, birthday, and a gender
class PatientRecord:
    def __init__(self, birthday, legal_name, birth_name, gender):
        self.birthday = birthday
        self.legal_name = legal_name
        self.birth_name = birth_name
        self.gender = gender

    @property
    def birthday(self):
        return self.__birthday

    @birthday.setter
    def birthday(self, birthday):
        self.__birthday = birthday

    @property
    def legal_name(self):
        return self.__legal_name

    @legal_name.setter
    def legal_name(self, legal_name):
        self.__legal_name = legal_name

    @property
    def birth_name(self):
        return self.__birth_name

    @birth_name.setter
    def birth_name(self, birth_name):
        self.__birth_name = birth_name

    @property
    def gender(self):
        return self.__gender

    @gender.setter
    def gender(self, gender):
        self.__gender = gender


# Method to parse out the name information and materialize it into an object
# Assumes the sequence of fields as is common in HL 2.2, 2.3.1, and 2.5.1
def extract_name_object(patient_name, hl7_msh_encoding_chars):
    patient_name_object = PatientName()

    patient_name_parts = patient_name.split(hl7_msh_encoding_chars[0])
    patient_name_parts_length = len(patient_name_parts)
    if patient_name_parts_length > 0:
        patient_name_object.family_name = patient_name_parts[0]
    if patient_name_parts_length > 1:
        patient_name_object.given_name = patient_name_parts[1]
    if patient_name_parts_length > 2:
        patient_name_object.middle_initial_or_name = patient_name_parts[2]
    if patient_name_parts_length > 3:
        patient_name_object.suffix = patient_name_parts[3]
    if patient_name_parts_length > 4:
        patient_name_object.prefix = patient_name_parts[4]

    return patient_name_object


# Parse ah hl7 PID with known field encoding characters into a materialized
# PatientRecord object
def parse_hl7_pid_to_patient_record(hl7_pid_segment, hl7_msh_encoding_chars,
    current_msh_version):
    segment_parts = hl7_pid_segment.split('|')

    # Pull out the patient name and normalize it
    patient_name = segment_parts[5]

    # For this exercise, MSH versions 2.2, 2.3.1, and 2.5.1 will be supported as
    # the data files seem to only contain this set
    patient_legal_name = None
    patient_birth_name = None
    if current_msh_version is None:
        current_msh_version = '2.5.1'
    if current_msh_version == '2.5.1':
        patient_names = patient_name.split(hl7_msh_encoding_chars[1])
        for patient_name_part in patient_names:
            patient_name_fields = patient_name_part.split(
                hl7_msh_encoding_chars[0])
            name_type = 'L'
            if len(patient_name_fields) > 6:
                name_type = patient_name_fields[6]
            if name_type == 'B':
                patient_birth_name = extract_name_object(patient_name_part,
                                                         hl7_msh_encoding_chars)
            else:
                patient_legal_name = extract_name_object(patient_name_part,
                                                         hl7_msh_encoding_chars)
    else:
        patient_legal_name = extract_name_object(patient_name,
                                                 hl7_msh_encoding_chars)

    # Parse out and normalize birthdays, if one is not set, then it will be
    # treated as None
    parsed_birthday = None
    if len(segment_parts[7]) > 0:
        parsed_birthday = datetime.strptime(segment_parts[7], "%Y%m%d")
    return PatientRecord(parsed_birthday, patient_legal_name,
                         patient_birth_name, segment_parts[8])


# Process all files recursively in the directly.  For this exercise the pulled
# out data is effectively the patient record.  Once the patient record is pulled
# out, then the general statistics are computed across both exercises for the
# homework
def hl7_data_files_processor(hl7_data_directory):
    # ensure path exists and is a directory
    if path.exists(hl7_data_directory) and path.isdir(hl7_data_directory):
        total_segments_processed = 0
        year_of_youngest_male = 0
        unique_female_patients = set()
        version_ids = set()
        # walk the path... (iterate over all the directories / files)
        for root, dirs, files in os.walk(path.abspath(hl7_data_directory)):
            for file in files:
                hl7_data_file = path.join(root, file)
                # ensure the target path is a file AND is readable
                if path.isfile(hl7_data_file) and os.access(hl7_data_file,
                                                            os.R_OK):
                    with open(hl7_data_file, 'r') as data_file:
                        # the current_separators list is updated once an MSH
                        # segment is encountered.  This set remains active until
                        # the next MSH segment is encountered at which point its
                        # updated and used from that point on
                        current_separators = []
                        current_version = None
                        for hl7_segment in data_file:
                            # Split the segment into into components
                            segment_parts = hl7_segment.split('|')
                            # if this is a Message Header (MSH), the update the
                            # segment parts
                            if segment_parts[0] == 'MSH':
                                current_separators = list(segment_parts[1])
                                current_version = segment_parts[11]
                                version_ids.add(current_version)
                            # if this is a PID segment, the parse it and pull
                            # out the relevant data.  Once that is done, and
                            # what is created is a materialized object, pull out
                            # the relevant data - birth year and uniqueness of
                            # female patients
                            if segment_parts[0] == 'PID':
                                patient_record = parse_hl7_pid_to_patient_record(
                                    hl7_segment,
                                    current_separators, current_version)
                                if patient_record.gender == 'M' and \
                                    patient_record.birthday is not None and \
                                    patient_record.birthday.year > year_of_youngest_male:
                                    year_of_youngest_male = patient_record.birthday.year
                                if patient_record.gender == 'F':
                                    if patient_record.legal_name is not None:
                                        unique_female_patients.add(
                                            str(patient_record.legal_name))
                                    if patient_record.birth_name is not None:
                                        unique_female_patients.add(
                                            str(patient_record.birth_name))

                            total_segments_processed += 1
        # Output the computed statistics
        print(f"Total Segments Processed: {total_segments_processed}")
        print(f"Youngest Male Patient: {year_of_youngest_male}")
        print(f"Num Female Patients: {len(unique_female_patients)}")
        print(f"Num Versions: {len(version_ids)} and they are: {version_ids}")
    else:
        raise Exception(
            f"Path {hl7_data_directory} does not seem to exist or is "
            f"not a directory")


if __name__ == '__main__':
    # Setup a parser to enable passing in command line arguments, right now
    # just one argument - the path to the directory containing the data to
    # process
    parser = argparse.ArgumentParser(description="Process HL7 Files")
    parser.add_argument('--hl7_data_directory', type=str, required=True,
                        help="Path to directory containing HL7 records, "
                             "it will be traversed recursively")
    args = parser.parse_args()
    hl7_data_files_processor(args.hl7_data_directory)
