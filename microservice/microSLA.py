# SLA Predictor application
# CLASS Project: https://class-project.eu/

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# Created on 25 Mar 2021
# @author: Jorge Montero - ATOS
#

from flask import Flask, request, render_template, jsonify

from prometheus_api_client import PrometheusConnect
from prometheus_api_client.utils import parse_datetime
from datetime import timedelta
import pandas as pd

import sklearn
import pickle as pk

prom = PrometheusConnect(url ="http://192.168.7.42:9091/", disable_ssl=True)

metrics_names = ['go_goroutines','go_memstats_alloc_bytes','go_memstats_gc_cpu_fraction',
'go_memstats_gc_sys_bytes', 'go_memstats_heap_alloc_bytes',
'go_memstats_heap_idle_bytes', 'go_memstats_heap_inuse_bytes',
'go_memstats_heap_objects', 'go_memstats_heap_released_bytes',
'go_memstats_heap_sys_bytes', 'go_memstats_last_gc_time_seconds',
'go_memstats_mspan_inuse_bytes', 'go_memstats_next_gc_bytes',
'go_memstats_other_sys_bytes','go_memstats_stack_inuse_bytes',
'go_memstats_stack_sys_bytes', 'go_threads', 'node_boot_time_seconds',
'node_entropy_available_bits', 'node_filefd_allocated' ,'node_load1',
'node_load15', 'node_load5', 'node_memory_Active_anon_bytes',
'node_memory_Active_bytes', 'node_memory_Active_file_bytes',
'node_memory_AnonHugePages_bytes', 'node_memory_AnonPages_bytes',
'node_memory_Buffers_bytes', 'node_memory_Cached_bytes',
'node_memory_Committed_AS_bytes', 'node_memory_DirectMap2M_bytes',
'node_memory_DirectMap4k_bytes', 'node_memory_Dirty_bytes',
'node_memory_Inactive_anon_bytes', 'node_memory_Inactive_bytes',
'node_memory_Inactive_file_bytes', 'node_memory_KernelStack_bytes',
'node_memory_Mapped_bytes', 'node_memory_MemAvailable_bytes',
'node_memory_MemFree_bytes', 'node_memory_PageTables_bytes',
'node_memory_SReclaimable_bytes', 'node_memory_SUnreclaim_bytes',
'node_memory_Shmem_bytes', 'node_memory_Slab_bytes', 'node_procs_running', 'node_sockstat_TCP_alloc',
'node_sockstat_TCP_mem', 'node_sockstat_TCP_mem_bytes',
'node_sockstat_sockets_used', 'node_time_seconds',
'node_timex_frequency_adjustment_ratio', 'node_timex_maxerror_seconds',
'node_timex_offset_seconds', 'process_resident_memory_bytes',
'process_start_time_seconds']


def get_timeseries_from_metric(metric_name,start_time,end_time,chunk_size):
    metric_data = prom.get_metric_range_data(
        metric_name,  # this is the metric name and label config
        start_time=start_time,
        end_time=end_time,
        chunk_size=chunk_size,
    )

    # do some process to it: merging all timeseries values to one, and get the aggregated value
    metric_d_all_df = pd.DataFrame()
    if metric_data:
        for i in range(0,len(metric_data)):
            metric_d_df = pd.DataFrame(metric_data[i]["values"],columns=["timestamp", metric_name+str(i)])
            metric_d_df['timestamp']= pd.to_datetime(metric_d_df['timestamp'], unit='s')
            metric_d_df[metric_name+str(i)]= pd.to_numeric(metric_d_df[metric_name+str(i)], errors='coerce')
            metric_d_df.set_index('timestamp', inplace=True)

            metric_d_all_df = pd.concat([metric_d_all_df, metric_d_df], axis=0)

        #metric_d_all_df = metric_d_all_df.groupby(pd.Grouper(freq='1Min')).aggregate("last")

        metric_d_agg_df = metric_d_all_df
        metric_d_agg_df[metric_name] = metric_d_all_df.aggregate("mean", axis=1)
        #return metric_d_agg_df[metric_name]

        metric_data_insert = []
        metric_data_insert_time = metric_d_agg_df.index.values
        metric_data_insert_val = metric_d_agg_df[metric_name].values
        for i in range(0,len(metric_data_insert_time)):
            metric_data_insert.append([metric_data_insert_time[i],metric_data_insert_val[i]])
        metric_data_df = pd.DataFrame(metric_data_insert,columns=["timestamp", metric_name])
        metric_data_df['timestamp']= pd.to_datetime(metric_data_df['timestamp'], unit='s')
        metric_data_df[metric_name]= pd.to_numeric(metric_data_df[metric_name], errors='coerce')
        metric_data_df.set_index('timestamp', inplace=True)
        return metric_data_df

    else:
        return pd.DataFrame()


def historical_values_for_metrics(start_time, end_time, chunk_size, metrics_names):
    metrics_all_df = pd.DataFrame()

    print(end_time)
    for metric_name in metrics_names:
        print(metric_name)
        metric_data_df = get_timeseries_from_metric(metric_name,start_time,end_time,chunk_size)
        #print(metric_data_df)
        if not metric_data_df.empty:
            metrics_all_df = pd.concat([metrics_all_df, metric_data_df], axis=0)

    metrics_all_df = metrics_all_df.groupby(pd.Grouper(freq='1Min')).aggregate("last")

    print("-------------------------------------")
    return metrics_all_df


def create_train(df_array,metrics_names):
    metrics_names_all = []
    metrics_names_all_data = []

    for metric_name in metrics_names:
        metrics_names_all.append(metric_name+"_mean")
        metrics_names_all.append(metric_name+"_max")
        metrics_names_all.append(metric_name+"_min")
        metrics_names_all.append(metric_name+"_std")

    for metrics_df in df_array:
        metrics_df_mean = metrics_df.aggregate("mean",axis=0)
        metrics_df_max = metrics_df.aggregate("max",axis=0)
        metrics_df_min = metrics_df.aggregate("min",axis=0)
        metrics_df_std = metrics_df.aggregate("std",axis=0)
        metrics_names_data = []
        for metric_name in metrics_names:
            metrics_names_data.append(metrics_df_mean[metric_name])
            metrics_names_data.append(metrics_df_max[metric_name])
            metrics_names_data.append(metrics_df_min[metric_name])
            metrics_names_data.append(metrics_df_std[metric_name])
        metrics_names_all_data.append(metrics_names_data)

    metrics_MLP = pd.DataFrame(data=metrics_names_all_data, columns=metrics_names_all)
    return metrics_MLP


def predictSLA(workers,exectime):
    start_time = parse_datetime("1h")
    end_time = parse_datetime("now")
    chunk_size = timedelta(minutes=10)

    w = [1,3,6,9]
    metrics_now = historical_values_for_metrics(start_time, end_time, chunk_size, metrics_names)
    metrics_MLP = create_train([metrics_now],metrics_names)

    best_model_reload = pk.load(open("best_model.pkl",'rb'))
    dataset = metrics_MLP.values
    x_input = dataset
    y_pred = best_model_reload.predict(x_input)

    inf=1000000
    status_workers_exectime = {}
    status_workers_exectime[1] = {}
    status_workers_exectime[1]["low"] = (550000,580000)
    status_workers_exectime[1]["normal"] = (580001,610000)
    status_workers_exectime[1]["high"] = (610001,inf)
    status_workers_exectime[3] = {}
    status_workers_exectime[3]["low"] = (550000,570000)
    status_workers_exectime[3]["normal"] = (570001,600000)
    status_workers_exectime[3]["high"] = (600001,inf)
    status_workers_exectime[6] = {}
    status_workers_exectime[6]["low"] = (430000,450000)
    status_workers_exectime[6]["normal"] = (450001,470000)
    status_workers_exectime[6]["high"] = (470001,inf)
    status_workers_exectime[9] = {}
    status_workers_exectime[9]["low"] = (390000,410000)
    status_workers_exectime[9]["normal"] = (410001,420000)
    status_workers_exectime[9]["high"] = (420001,inf)

    estimated_exectimes = []
    for i in w:
        if exectime >= status_workers_exectime[i][y_pred[0]][0]:
            estimated_exectimes.append((i,status_workers_exectime[i][y_pred[0]]))

    return estimated_exectimes

def filterSLA(workers,resultSLA):
    filteredSLA = -1
    if workers in resultSLA:
        filteredSLA = workers
    else:
        diffworker = 100
        for slaworker in resultSLA:
            if abs(workers-slaworker[0]) < diffworker:
                diffworker = abs(workers-slaworker[0])
                filteredSLA = slaworker[0]

    return filteredSLA



app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predictSLA', methods=['GET'])
def modelPredictSLA():
    workers = int(request.args.get('workers'))
    exectime = int(request.args.get('exectime'))
    resultSLA = predictSLA(workers,exectime)
    print(resultSLA)
    filteredSLA = filterSLA(workers,resultSLA)
    return str(filteredSLA)

if __name__ == '__main__':
    app.run(host=os.environ.get('HTTP_HOST', '0.0.0.0'),
        port=int(os.environ.get('HTTP_PORT', '5002')))

