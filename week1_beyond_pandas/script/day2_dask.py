#!/usr/bin/env python
# coding: utf-8

# ## References
# 
# - https://docs.dask.org/

from IPython.display import Image


# ## Single Machine: dask.distributed

from dask.distributed import Client
client = Client()
client


# ## dask-scheduler
# 
# A dask.distributed network consists of one dask-scheduler process and several dask-worker processes that connect to that scheduler. These are normal Python processes that can be executed from the command line. We launch the dask-scheduler executable in one process and the dask-worker executable in several processes, possibly on different machines
# 
# https://docs.dask.org/en/latest/setup/cli.html
# 
# <img src="img/Screen Shot 2021-01-03 at 00.37.38.png">

# ## Why Dask?
# 
# - It provide method to scale up Pandas, Scikit-Learn, and Numpy with native feature and minimal rewriting
# - Dask can enable efficient parallel computations on single machines by leveraging their multi-core CPUs and streaming data efficiently from disk [1]
# - Smaller and lighter weight than Spark [2]
# 
# 
# [1] https://docs.dask.org/en/latest/why.html
# 
# [2] https://docs.dask.org/en/latest/spark.html

# ## DataFrames: Read and Write Data

# from dask.distributed import Client
# client = Client(n_workers=1, threads_per_worker=6, processes=False, memory_limit='60GB')
# client


cell_ref_path = "data/cell.csv"
event_input_path = "data/data_1gb.csv"
output_path = "output/mapped_cells_1gb.csv"


# ### Create output in HDF5

get_ipython().run_cell_magic('time', '', 'import dask.dataframe as dd\ncell_input = dd.read_csv(cell_ref_path)\nevent_input = dd.read_csv(event_input_path)\n\nmapped_cells = dd.merge(event_input,cell_input,left_on=\'mno_cell_id\',right_on=\'mno_cell_id\').compute()\n\nmapped_cells = mapped_cells[[\'mno_ms_id\',\'pos_time\',\'cell_id\']]\nmapped_cells = mapped_cells.rename(columns={\'cell_id\': \'mno_cell_id\'})\n\nmapped_cells.to_hdf(output_path + ".hdf5", \'/data\')')


# ### Create output in CSV

get_ipython().run_cell_magic('time', '', "import dask.dataframe as dd\ncell_input = dd.read_csv(cell_ref_path)\nevent_input = dd.read_csv(event_input_path)\n\nmapped_cells = dd.merge(event_input,cell_input,left_on='mno_cell_id',right_on='mno_cell_id').compute()\n\nmapped_cells = mapped_cells[['mno_ms_id','pos_time','cell_id']]\nmapped_cells = mapped_cells.rename(columns={'cell_id': 'mno_cell_id'})\n\nmapped_cells.to_csv(output_path, index=False)")


# ## Conclusion
# 
# - In Dask, writing an output file in HDF5 was much faster than CSV file 
# - Without any much setup, Dask already doing the data processing in multi-processing
