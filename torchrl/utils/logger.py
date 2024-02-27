import tensorboardX
import logging
import shutil
import os
import numpy as np
import sys
import json5 as json
import csv

class Logger():
    def __init__(self, experiment_id, env_name, alg_name, seed, log_time, params, log_dir = "./log"):

        self.logger = logging.getLogger("{}_{}_{}_{}".format(experiment_id,env_name,alg_name,str(seed)))

        self.logger.handlers = []
        self.logger.propagate = False
        sh = logging.StreamHandler(sys.stdout)
        format = "%(asctime)s %(threadName)s %(levelname)s: %(message)s"
        formatter = logging.Formatter(format)
        sh.setFormatter(formatter)
        sh.setLevel(logging.INFO)
        self.logger.addHandler( sh )
        self.logger.setLevel(logging.INFO)

        work_dir = os.path.join( log_dir, experiment_id, alg_name, log_time + '-' + str(seed) )
        self.work_dir = work_dir
        if os.path.exists( work_dir ):
            shutil.rmtree(work_dir)
        self.tf_writer = tensorboardX.SummaryWriter(work_dir)

        self.csv_file_path = os.path.join(work_dir, 'log.csv')

        self.update_count = 0
        self.stored_infos = {}

        with open( os.path.join(work_dir, 'params.json'), 'w' ) as output_param:
            json.dump( params, output_param, indent = 2 )

        self.logger.info("Experiment Name:{}".format(experiment_id))
        self.logger.info(
            json.dumps(params, indent = 2 )
        )

    def log(self, info):
        self.logger.info(info)

    def add_update_info(self, infos):

        for info in infos:
            self.stored_infos[info] = infos[info]
            
        self.update_count += 1
    
    def add_epoch_info(self, epoch_num, total_frames, total_time, infos, csv_write=True):
        if csv_write:
            if epoch_num == 0:
                self.csv_titles = ["EPOCH", "Time Consumed", "Total Frames"]
                csv_values = [epoch_num, total_time, total_frames]
            else:
                csv_values = [epoch_num, total_time, total_frames] + [""] * (len(self.csv_titles) - 3)

                
        self.logger.info("EPOCH:{}".format(epoch_num))
        # self.logger.info("Time Consumed:{}s".format(total_time))
        # self.logger.info("Total Frames:{}s".format(total_frames))
        
        for info in infos:
            self.tf_writer.add_scalar(info, infos[info], total_frames)
            if csv_write:
                if epoch_num == 0:
                    self.csv_titles += [info]
                    csv_values += ["{:.5f}".format(infos[info])]
                else:
                    csv_values[self.csv_titles.index(info)] = "{:.5f}".format(infos[info])

        for info in self.stored_infos:
            self.tf_writer.add_scalar(info, self.stored_infos[info], total_frames )
            if csv_write:
                if epoch_num == 0:
                    self.csv_titles += [info]
                    csv_values += ["{:.5f}".format(self.stored_infos[info])]
                else:
                    csv_values[self.csv_titles.index(info)] = "{:.5f}".format(self.stored_infos[info])
        #clear
        self.stored_infos = {}
        if csv_write:
            with open(self.csv_file_path, 'a') as f:
                self.csv_writer = csv.writer(f)
                if epoch_num == 0:
                    self.csv_writer.writerow(self.csv_titles)
                self.csv_writer.writerow(csv_values)
    
    def log_reset_task_num(self, reset_num, step):
        with open( os.path.join(self.work_dir, 'reset_cnt.txt'), 'a' ) as f:
            f.write(str(step) + ": " + str(reset_num) + "\n")
    
    def close(self):
        self.tf_writer.close()


class EvalLogger():
    def __init__(self, experiment_id, env_name, alg_name, seed, log_time, log_dir = "./log"):

        self.logger = logging.getLogger("{}_{}_{}_{}".format(experiment_id,env_name,alg_name,str(seed)))

        self.logger.handlers = []
        self.logger.propagate = False
        sh = logging.StreamHandler(sys.stdout)
        format = "%(asctime)s %(threadName)s %(levelname)s: %(message)s"
        formatter = logging.Formatter(format)
        sh.setFormatter(formatter)
        sh.setLevel(logging.INFO)
        self.logger.addHandler( sh )
        self.logger.setLevel(logging.INFO)

        work_dir = os.path.join( log_dir, experiment_id, alg_name, log_time + '-' + str(seed))
        self.work_dir = work_dir
        self.tf_writer = tensorboardX.SummaryWriter(work_dir)

        self.csv_file_path = os.path.join(work_dir, 'test_log.csv')

        self.update_count = 0
        self.stored_infos = {}

    def log(self, info):
        self.logger.info(info)

    def add_update_info(self, infos):
        for info in infos:
            self.stored_infos[info] = infos[info]
            
        self.update_count += 1
    
    def add_epoch_eval_info(self, epoch_num, total_frames, total_time, infos, csv_write=True):
        if csv_write:
            if epoch_num == 0:
                self.csv_titles = ["EPOCH", "Time Consumed", "Total Frames"]
                csv_values = [epoch_num, total_time, total_frames]
            else:
                csv_values = [epoch_num, total_time, total_frames] + [""] * (len(self.csv_titles) - 3)

                
        self.logger.info("EPOCH Test:{}".format(epoch_num))
        # self.logger.info("Time Consumed Test:{}s".format(total_time))
        # self.logger.info("Total Frames:{}s".format(total_frames))
        
        for info in infos:
            self.tf_writer.add_scalar(info, infos[info], total_frames)
            if csv_write:
                if epoch_num == 0:
                    self.csv_titles += [info]
                    csv_values += ["{:.5f}".format(infos[info])]
                else:
                    csv_values[self.csv_titles.index(info)] = "{:.5f}".format(infos[info])

        for info in self.stored_infos:
            self.tf_writer.add_scalar(info, self.stored_infos[info], total_frames)
            if csv_write:
                if epoch_num == 0:
                    self.csv_titles += [info]
                    csv_values += ["{:.5f}".format(self.stored_infos[info])]
                else:
                    csv_values[self.csv_titles.index(info)] = "{:.5f}".format(self.stored_infos[info])
        
        self.stored_infos = {}
        if csv_write:
            with open(self.csv_file_path, 'a') as f:
                self.csv_writer = csv.writer(f)
                if epoch_num == 0:
                    self.csv_writer.writerow(self.csv_titles)
                self.csv_writer.writerow(csv_values)
    
    def close(self):
        self.tf_writer.close()
    
    def save_gates_info(self, gates_name, gates_info):
        for i, gates_info_i in enumerate(gates_info):
            file_name = 'task_' + str(i) + '_gates_info.csv'
            gates_info_i = np.around(gates_info_i, decimals=2)
            with open( os.path.join(self.work_dir, file_name), 'a' ) as f:
                f.write(gates_name + ", ")
                f.write(str(list(gates_info_i))[1:-1] + "\n")
            
    
