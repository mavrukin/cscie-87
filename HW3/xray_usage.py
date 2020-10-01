import argparse
import csv
from datetime import datetime
from datetime import timedelta
from datetime import time
import os
import os.path as path


# A class that represents counting statistics for optimal utilization of wait
# and exam rooms.
class XRayRoomsBucketStats:
    def __init__(self):
        self.peak_rooms = 0
        self.current_rooms = 0
        self.wait_queue = 0
        self.peak_wait_queue = 0
        self.waiting_patients = set()
        self.exam_patients = set()

    @property
    def peak_rooms(self):
        return self.__peak_rooms

    @property
    def current_rooms(self):
        return self.__current_rooms

    @property
    def wait_queue(self):
        return self.__wait_queue

    @property
    def peak_wait_queue(self):
        return self.__peak_wait_queue

    @property
    def num_waiting_patients(self):
        return len(self.__waiting_patients)

    @property
    def waiting_patients(self):
        return self.__waiting_patients

    @waiting_patients.setter
    def waiting_patients(self, patients):
        self.__waiting_patients = patients

    def add_waiting_patient(self, patient_id):
        self.__waiting_patients.add(patient_id)

    @property
    def num_exam_patients(self):
        return len(self.__exam_patients)

    @property
    def exam_patients(self):
        return self.__exam_patients

    @exam_patients.setter
    def exam_patients(self, patients):
        self.__exam_patients = patients

    def add_exam_patient(self, patient_id):
        self.__exam_patients.add(patient_id)

    @peak_rooms.setter
    def peak_rooms(self, peak_rooms):
        self.__peak_rooms = peak_rooms

    @current_rooms.setter
    def current_rooms(self, current_rooms):
        self.__current_rooms = current_rooms

    @wait_queue.setter
    def wait_queue(self, wait_queue):
        self.__wait_queue = wait_queue

    @peak_wait_queue.setter
    def peak_wait_queue(self, peak_wait_queue):
        self.__peak_wait_queue = peak_wait_queue

    def add_room(self):
        self.current_rooms += 1
        if self.current_rooms > self.peak_rooms:
            self.peak_rooms = self.current_rooms

    def remove_room(self):
        self.current_rooms -= 1

    def add_wait(self):
        self.wait_queue += 1
        if self.wait_queue > self.peak_wait_queue:
            self.peak_wait_queue = self.wait_queue

    def remove_wait(self):
        self.wait_queue -= 1


def date_str_to_datetime(date_str):
    return datetime.strptime(date_str, "%m/%d/%y %H:%M")


def process_xray_data_file(xray_csv_file):
    if path.isfile(xray_csv_file) and os.access(xray_csv_file, os.R_OK):
        row_counter = 0
        patient_frequency = dict()
        xray_room_buckets = dict()
        earliest_xray_begin_time = time.max
        latest_xray_begin_time = time.min
        earliest_xray_end_time = time.max
        latest_xray_end_time = time.min
        seven_nine_counter = set()
        ten_twelve_counter = set()
        thirteen_fifteen_counter = set()
        fourteen_sixteen_counter = set()
        with open(xray_csv_file) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                row_counter += 1
                patient_id = row["stMRN"]
                if patient_id not in patient_frequency:
                    patient_frequency[patient_id] = 0
                patient_frequency[patient_id] += 1
                # Time Boundaries: [hour), [minute) - thus:
                # 7:29 will be in the 7:00 bucket but 7:30 will be in the 7.5
                # bucket
                dtArrive = date_str_to_datetime(row["dtArrive"])
                dtBegin = date_str_to_datetime(row["dtBegin"])
                dtCompleted = date_str_to_datetime(row["dtCompleted"])

                if 7 <= dtBegin.time().hour < 9 or 7 <= dtCompleted.time().hour < 9:
                    seven_nine_counter.add(patient_id)
                if 10 <= dtBegin.time().hour < 12 or 10 <= dtCompleted.time().hour < 12:
                    ten_twelve_counter.add(patient_id)
                if 13 <= dtBegin.time().hour < 15 or 13 <= dtCompleted.time().hour < 15:
                    thirteen_fifteen_counter.add(patient_id)
                if 14 <= dtBegin.time().hour < 16 or 14 <= dtCompleted.time().hour < 16:
                    fourteen_sixteen_counter.add(patient_id)

                if dtBegin.time() < earliest_xray_begin_time:
                    earliest_xray_begin_time = dtBegin.time()
                if dtBegin.time() > latest_xray_begin_time:
                    latest_xray_begin_time = dtBegin.time()

                if dtCompleted.time() < earliest_xray_end_time:
                    earliest_xray_end_time = dtBegin.time()
                if dtCompleted.time() > latest_xray_end_time:
                    latest_xray_end_time = dtBegin.time()

                while dtArrive <= dtBegin:
                    if dtArrive not in xray_room_buckets:
                        xray_room_buckets[dtArrive] = XRayRoomsBucketStats()
                    xray_room_buckets[dtArrive].add_wait()
                    xray_room_buckets[dtArrive].add_waiting_patient(
                        row["stMRN"])
                    dtArrive = dtArrive + timedelta(minutes=+1)

                if dtBegin not in xray_room_buckets:
                    xray_room_buckets[dtBegin] = XRayRoomsBucketStats()
                xray_room_buckets[dtBegin].remove_wait()

                while dtBegin <= dtCompleted:
                    if dtBegin not in xray_room_buckets:
                        xray_room_buckets[dtBegin] = XRayRoomsBucketStats()
                    xray_room_buckets[dtBegin].add_room()
                    xray_room_buckets[dtBegin].add_exam_patient(row["stMRN"])
                    dtBegin = dtBegin + timedelta(minutes=+1)

                if dtCompleted not in xray_room_buckets:
                    # Shouldn't happen - error in data
                    print(
                        f"dtCompleted data error at row [{row_counter}] on date {dtCompleted}")
                    xray_room_buckets[dtCompleted] = XRayRoomsBucketStats()
                xray_room_buckets[dtCompleted].remove_room()
            else:
                print(
                    f"Bad data arrive {dtArrive}, begin {dtBegin}, complete {dtCompleted}")

        print(
            f"{row_counter} total rows were read with {len(patient_frequency)}"
            f" patients")
        print(f"The patient that visited the most: "
              f"{max(patient_frequency, key=patient_frequency.get)} with a total"
              f" of {max(patient_frequency.values())} visits")
        print(
            f"Earliest XRay Begin found time: {earliest_xray_begin_time} latest found time: {latest_xray_begin_time}")
        print(
            f"Earliest XRay End found time: {earliest_xray_end_time} latest found time: {latest_xray_end_time}")

        peak_room_usage = XRayRoomsBucketStats()
        peak_room_date = datetime.now()
        peak_wait = XRayRoomsBucketStats()
        peak_wait_date = datetime.now()
        peak_wait_set = 0
        peak_exam_set = 0
        for exam_date, room_stats in xray_room_buckets.items():
            if room_stats.peak_rooms > peak_room_usage.peak_rooms:
                peak_room_usage = room_stats
                peak_room_date = exam_date
            if room_stats.peak_wait_queue > peak_wait.peak_wait_queue:
                peak_wait = room_stats
                peak_wait_date = exam_date
            if room_stats.num_waiting_patients > peak_wait_set:
                peak_wait_set = room_stats.num_waiting_patients
            if room_stats.num_exam_patients > peak_exam_set:
                peak_exam_set = room_stats.num_exam_patients
        print(f"Peak Usage: {peak_room_usage.peak_rooms} on {peak_room_date}")
        print(f"Peak Wait: {peak_wait.peak_wait_queue} on {peak_wait_date}")
        print(f"Peak Wait Set: {peak_wait_set}, Peak Exam Set: {peak_exam_set}")
        print(f"7:00 - 9:00: {len(seven_nine_counter)}")
        print(f"10:00 - 12:00: {len(ten_twelve_counter)}")
        print(f"13:00 - 15:00: {len(thirteen_fifteen_counter)}")
        print(f"14:00 - 16:00: {len(fourteen_sixteen_counter)}")
    else:
        raise Exception(f"{path.abspath(xray_csv_file)} does not seem to be a "
                        f"valid path or it is not readable")


# This method does the main sampling of the results
# start_slot - the start time from the event log (type: datetime)
# end_slot - the end time from the event log (type: datetime)
# sample_counter - a map where keys are rounded sample time (0/:30) with values
#                  being a set of patient ids (type: [datetime --> set(str)])
#                  This parameter is in/out - will be updated within the method
# patient_id - the unique id of the patient associated with this even
def sample_time_slots_and_update_tracker(start_slot, end_slot, samples_counter,
    patient_id):
    if start_slot < end_slot:
        if start_slot.year == end_slot.year \
            and start_slot.month == end_slot.month \
            and start_slot.day == end_slot.day \
            and (start_slot.hour < end_slot.hour
                 or (start_slot.minute < 30 <= end_slot.minute)):
            while start_slot < end_slot:
                wait_observation_time = datetime(
                    start_slot.year, start_slot.month, start_slot.day,
                    (start_slot.hour + 1 if start_slot.minute >= 30
                     else start_slot.hour),
                    (0 if start_slot.minute >= 30 else 30))
                if wait_observation_time not in samples_counter:
                    samples_counter[wait_observation_time] = set()
                samples_counter[wait_observation_time].add(patient_id)
                start_slot = start_slot + timedelta(minutes=+30)


def process_xray_data_sampling(xray_csv_file):
    if path.isfile(xray_csv_file) and os.access(xray_csv_file, os.R_OK):
        row_counter = 0
        wait_samples = dict()
        exam_samples = dict()
        patient_frequency = dict()
        with open(xray_csv_file) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                row_counter += 1
                patient_id = row["stMRN"]
                if patient_id not in patient_frequency:
                    patient_frequency[patient_id] = 0
                patient_frequency[patient_id] += 1
                # Time Boundaries: [hour), [minute) - thus:
                # 7:29 will be in the 7:00 bucket but 7:30 will be in the 7.5
                # bucket
                dt_arrive = date_str_to_datetime(row["dtArrive"])
                dt_begin = date_str_to_datetime(row["dtBegin"])
                dt_completed = date_str_to_datetime(row["dtCompleted"])

                sample_time_slots_and_update_tracker(
                    dt_arrive, dt_begin, wait_samples, patient_id)
                sample_time_slots_and_update_tracker(
                    dt_begin, dt_completed, exam_samples, patient_id)

        print(f"Max Wait Queue: {max(map(len, wait_samples.values()))}")
        print(f"Max Exam Rooms Used: {max(map(len, exam_samples.values()))}")


if __name__ == '__main__':
    # Setup a parser to enable passing in command line arguments, right now
    # just one argument - the path to the directory containing the data to
    # process
    parser = argparse.ArgumentParser(description="Process XRay Utilization")
    parser.add_argument('--xray_data_file', type=str, required=True,
                        help="Path to csv containing xray records")
    args = parser.parse_args()
    print("Exact Counting Results: ")
    print("------------------------")
    process_xray_data_file(args.xray_data_file)
    print("------------------------")
    print("Sampling Approach: ")
    print("------------------------")
    process_xray_data_sampling(args.xray_data_file)
