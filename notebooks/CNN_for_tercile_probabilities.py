#!/usr/bin/env python
# coding: utf-8

# # Train ML model to correct predictions of week 3-4 & 5-6
# 
# This notebook create a Machine Learning `ML_model` to predict weeks 3-4 & 5-6 based on `S2S` weeks 3-4 & 5-6 forecasts and is compared to `CPC` observations for the [`s2s-ai-challenge`](https://s2s-ai-challenge.github.io/).

# https://s2s-ai-challenge.github.io/
# 
# We deal with two fundamentally different variables here: 
# - Total precipitation is precipitation flux pr accumulated over lead_time until valid_time and therefore describes a point observation. 
# - 2m temperature is averaged over lead_time(valid_time) and therefore describes an average observation. 
# 
# The submission file data model unifies both approaches and assigns 14 days for week 3-4 and 28 days for week 5-6 marking the first day of the biweekly aggregate.

#some ideas for improvements:
#circular input
#https://www.tu-chemnitz.de/etit/proaut/publications/schubert19_IV.pdf
#https://www.tu-chemnitz.de/etit/proaut/en/research/ccnn.html
#add indices
#https://stackoverflow.com/questions/47818968/adding-an-additional-value-to-a-convolutional-neural-network-input
#https://datascience.stackexchange.com/questions/68450/how-can-you-include-information-not-present-in-an-image-for-neural-networks
#https://www.nature.com/articles/s41598-019-42294-8



# In[1]:

#do this before you import numpy if you run on the server. but not sure whether it really does anything
# =============================================================================
# import os
# os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
# os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6
# 
# import psutil
# psutil.Process().nice(19)# if on *ux
# =============================================================================
#import sys
#sys.path.insert(0,"..")



from tensorflow.keras.layers import Input, Dense, Flatten
#from tensorflow.keras.models import Sequential
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Reshape, Dot, Add, Activation
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import xarray as xr
xr.set_options(display_style='text')

import xskillscore as xs

#get_ipython().run_line_magic('matplotlib', 'inline')
#so that figures appear again

from scripts import make_probabilistic
from scripts import add_valid_time_from_forecast_reference_time_and_lead_time
from scripts import skill_by_year
from scripts import add_year_week_coords


from helper_ml_data import load_data, get_basis, rm_annualcycle, rm_tercile_edges, DataGenerator, single_prediction, skill_by_year_single


import warnings
warnings.simplefilter("ignore")

#server
# =============================================================================
# try:
#     import mkl
#     mkl.set_num_threads(4)
# except:
#     pass
# =============================================================================


# In[2]:
### set see to make shuffling in datagenerator reproducible
np.random.seed(42)

# # Get training data
# 
# preprocessing of input data may be done in separate notebook/script

# In[3]:


path_data = 'local_n'#'server'#
#['../../../../Data/s2s_ai/data', '../../../../Data/s2s_ai']


# ## Hindcast
hind_2000_2019_raw = load_data(data = 'hind_2000-2019', aggregation = 'biweekly', path = path_data,var_list = ['t2m'])#['tp','t2m','sm20','sst'])#,'lsm', 'msl', var_list = ['tp','t2m'])#path_data)
if 'sm20' in hind_2000_2019_raw.keys():
    hind_2000_2019_raw = hind_2000_2019_raw.isel(depth_below_and_layer = 0 ).reset_coords('depth_below_and_layer', drop=True)
elif 'msl' in hind_2000_2019_raw.keys():
    hind_2000_2019_raw = hind_2000_2019_raw.isel(meanSea = 0).reset_coords('meanSea', drop = True)


# ## Observations corresponding to hindcasts
obs_2000_2019_raw = load_data(data = 'obs_2000-2019', aggregation = 'biweekly', path = path_data)
# terciled
obs_2000_2019_terciled_raw = load_data(data = 'obs_terciled_2000-2019', aggregation = 'biweekly', path = path_data)

hind_2000_2019_raw = hind_2000_2019_raw[['t2m']]
# In[4]:

#lists to store results of optimization experiments
setup_params = []
acc_per_fold = [] 
loss_per_fold = []
skill_year1 = []
skill_year2 = []
percent_min = []
percent_max = []
history = []


# ### Select region
# 
# * use the whole globe as input since global teleconnections are a major source of predictability, at least over Europe.
# * create predictions for a much smaller domain, since otherwise the basis functions (and the associated multiplication) uses too much memory.
# 
# to make life easier for the beginning --> no periodic padding needed.

# =============================================================================
# #Does a larger output domain help?        
# input_lats = [slice(70,40), slice(90,30), slice(85.5,25), slice(90,-90)]#20,41,41,121
# input_lons = [slice(60,90), slice(0,90), slice(30,120), slice(0, 360)]#21,61,61,240
# 
# output_lats = [slice(60,50), slice(60,50),slice(60,50), slice(60,50)]#slice(90,0)]#7,7,7,7###,61
# output_lons = [slice(70,80), slice(70,80), slice(70,80), slice(70,80)]#slice(0,90)]#7,7,7,7###,61
# =============================================================================

#How large can I make my output domain?
input_lats = [slice(60,55), slice(63,52),slice(69,46), slice(81,34)]
input_lons = [slice(70,75), slice(67,78), slice(61,85), slice(49,97)]

output_lats = [slice(60,55), slice(63,52),slice(69,46), slice(81,34)]#4, 8, 16, 32
output_lons = [slice(70,75), slice(67,78), slice(61,85), slice(49,97)]
#small, middle, middle with centered output region, large

for i  in  range(0, len(input_lats)):
    print(i)
    input_lon = input_lons[i]
    input_lat = input_lats[i]
    
    
    output_lon = output_lons[i]
    output_lat = output_lats[i]

    rad = [4,8,16,32]#[4,6,7,10]
    basis_rad = rad[i]
    #for basis_rad in rad:
# =============================================================================
# input_lat = slice(70,40)#slice(90,-20)
# input_lon = slice(60,90) #slice(0,90)  slice(300,360)#negative,positive will not work
# 
# output_lat = slice(60,50)
# output_lon = slice(70,80)
# =============================================================================

    
    hind_2000_2019 = hind_2000_2019_raw.sel(longitude = input_lon, latitude = input_lat)
    obs_2000_2019 = obs_2000_2019_raw.sel(longitude = output_lon, latitude = output_lat)
    obs_2000_2019_terciled = obs_2000_2019_terciled_raw.sel(longitude = output_lon, latitude = output_lat)
    
    #obs_2000_2019.t2m.isel(lead_time = 0, forecast_time = 0).plot()
    
    # In[5]:
    # ### define some vars
    
    v='t2m'
    
    
    # 2 bi-weekly `lead_time`: week 3-4
    lead_input = hind_2000_2019.lead_time[0]
    lead_output = obs_2000_2019.lead_time[0]
    
    # In[6]:
    
    # ## masking: only used for label data
    #mask: same missing values at all forecast_times
    mask = xr.where(obs_2000_2019.notnull(),1,np.nan).mean('forecast_time', skipna = False)
    #mask.isel(lead_time = 0).t2m.plot()
    #What if hind contains nan?--> precip
    
    #############################include from here to optimise radius of basis functions
    
    # In[7]:
    # #### compute basis
    
    basis, lats, lons, n_xy, n_basis = get_basis(obs_2000_2019_terciled[v], basis_rad) #30#14
    ##the smaller you choose the radius of the basis functions, the more memory needs to be allocated (a lot more!)
    
    #basis.shape
    
    
    # In[8]:
    # ## climatology
    
    n_bins = 3 #we have to predict probs for 3 bins
    clim_probs = np.log(1/3)*np.ones((n_xy,n_bins))
    
    
    # In[8]:
    # ## Train Validation split
    
    # =============================================================================
    # # time is the forecast_time
    # time_train_start,time_train_end='2000','2017' # train
    # time_valid_start,time_valid_end='2018','2019' # valid
    # 
    # #train
    # fct_train = hind_2000_2019.sel(forecast_time=slice(time_train_start,time_train_end))#[v]
    # verif_train = obs_2000_2019_terciled.sel(forecast_time=slice(time_train_start,time_train_end))[v]
    # 
    # #fct_train = fct_train.where(mask[v].notnull())
    # verif_train = verif_train.where(mask[v].notnull())
    # 
    # #validation
    # fct_valid = hind_2000_2019.sel(forecast_time=slice(time_valid_start,time_valid_end))#[v]
    # verif_valid = obs_2000_2019_terciled.sel(forecast_time=slice(time_valid_start,time_valid_end))[v]
    # 
    # #fct_valid = fct_valid.where(mask[v].notnull())
    # verif_valid = verif_valid.where(mask[v].notnull())
    # =============================================================================
    
    
    # In[9]:
    
    ###############################include here to cross-val to avoid data leakage
    ###crossval 
    #num_folds = 10
    # Define per-fold score containers

    
    #standardization = True
    #rmcycle = True
    #for standardization in [True, False]:
    #    for rmcycle in [True, False]:
    
    standardization = True
    rmcycle = True
    rmterciles = True

    # K-fold Cross Validation model evaluation
    #fold_no = 0
    for fold_no in range(0,2):#range(0,10):
        print('fold {}'.format(fold_no))
    
        valid_indeces = range(fold_no*2*53, fold_no*2*53 + 53*2)
        train_indeces = [i for i in range(53*20) if i not in valid_indeces]
        #for i in train_indeces:
        #    del train_indeces[i]
        
        # =============================================================================
        # folds = [[slice('2000','2001'), slice('2002','2019')],
        #          [slice('2002','2003'), slice('2002','2019')],
        #          [slice('2000','2001'), slice('2002','2019')],
        #          [slice('2000','2001'), slice('2002','2019')],
        #          [slice('2000','2001'), slice('2002','2019')],
        #          [slice('2000','2001'), slice('2002','2019')],
        #          [slice('2000','2001'), slice('2002','2019')],
        #          [slice('2000','2001'), slice('2002','2019')],
        #          [slice('2000','2001'), slice('2002','2019')]]
        # =============================================================================
        
        # In[10]:
        # ### create datasets
        
        #train
        fct_train = hind_2000_2019.isel(forecast_time=train_indeces)#[v]
          
        verif_train = obs_2000_2019_terciled.isel(forecast_time=train_indeces)[v]
        verif_train = verif_train.where(mask[v].notnull())
          
        #validation
        fct_valid = hind_2000_2019.isel(forecast_time=valid_indeces)
        
        verif_valid = obs_2000_2019_terciled.isel(forecast_time=valid_indeces)[v]
        verif_valid = verif_valid.where(mask[v].notnull())
        
        
        
        # In[10]:
        # ### create datasets
        
        # =============================================================================
        # #train
        # fct_train = hind_2000_2019.sel(forecast_time=slice(time_train_start,time_train_end))#[v]
        # verif_train = obs_2000_2019_terciled.sel(forecast_time=slice(time_train_start,time_train_end))[v]
        # 
        # #fct_train = fct_train.where(mask[v].notnull())
        # verif_train = verif_train.where(mask[v].notnull())
        # 
        # #validation
        # fct_valid = hind_2000_2019.sel(forecast_time=slice(time_valid_start,time_valid_end))#[v]
        # verif_valid = obs_2000_2019_terciled.sel(forecast_time=slice(time_valid_start,time_valid_end))[v]
        # 
        # #fct_valid = fct_valid.where(mask[v].notnull())
        # verif_valid = verif_valid.where(mask[v].notnull())
        # =============================================================================
        
        
        # In[11]:
        # ### input features
        
        #use ensemble mean as input
        fct_train = fct_train.mean('realization')#.isel(realization = 0)
        fct_valid = fct_valid.mean('realization')#.isel(realization = 0)
        
        # In[12]:
        # ### Remove annual cycle
        
        ### dont remove annual cycle from land-sea mask!
        # ### land sea mask varies over time and ensemble members!
        if rmcycle == True:
            var_list = [i for i in fct_train.keys() if i not in ['lsm',v]]
            
            if len(var_list) == 1:
                fct_train[var_list] = rm_annualcycle(fct_train[var_list], fct_train[var_list])[var_list[0]]
                fct_valid[var_list] = rm_annualcycle(fct_valid[var_list], fct_valid[var_list])[var_list[0]]
            elif len(var_list) >1:
                fct_train[var_list] = rm_annualcycle(fct_train[var_list], fct_train[var_list])
                fct_valid[var_list] = rm_annualcycle(fct_valid[var_list], fct_valid[var_list])
                
        if rmterciles == True: 
            tercile_edges = load_data(data = 'obs_tercile_edges_2000-2019', aggregation = 'biweekly', path = path_data)
            #using tercile edges as input could be considered as data leakage during training between train and val 
            deb = rm_tercile_edges(fct_train[v], tercile_edges[v]).assign_coords( category_edge = ['lower_{}'.format(v),'upper_{}'.format(v)]).to_dataset(dim ='category_edge')
            fct_train = xr.merge([fct_train, deb])
            fct_train = fct_train.drop_vars(v)
            
            deb = rm_tercile_edges(fct_valid[v], tercile_edges[v]).assign_coords( category_edge = ['lower_{}'.format(v),'upper_{}'.format(v)]).to_dataset(dim ='category_edge')
            fct_valid = xr.merge([fct_valid, deb])
            fct_valid = fct_valid.drop_vars(v)
        # In[13]:
        #standardization:
        
        #std over forecast_time
        #fct_train_std.msl.mean('realization').isel(lead_time = 0).plot()
        #don't divide by local standard diviation!
        
        #fct_train.sel(lead_time = lead_input).isel(forecast_time = 0)['msl'].plot()#'t2m' 
        # t2m -10 bis 10, tp: -150 bis 150, sm20: -200 bis 200 --> maybe some rescaling and cutting of the outliers would help, sst: -3 bis 3, slm: 0 bis 1, msl: -1500 bis 1500.
        
        if standardization == True:
            #normalisation constant for each field
            fct_train_std_spatial = fct_train.std(('latitude','longitude')).mean(('forecast_time'))#mean over 'realization', already computed : this is the spatial standard deviation of the ensemble mean
            
            #do the normalization
            fct_train = fct_train / fct_train_std_spatial
            fct_valid = fct_valid / fct_train_std_spatial#use train set for standardization
            
        
        # =============================================================================
        # print(fct_train_std_spatial.isel(lead_time = 0 ).t2m.values)
        # print(fct_train_std_spatial.isel(lead_time = 0 ).tp.values)
        # print(fct_train_std_spatial.isel(lead_time = 0 ).sm20.values)
        # print(fct_train_std_spatial.isel(lead_time = 0 ).sst.values)
        # print(fct_train_std_spatial.isel(lead_time = 0 ).lsm.values)
        # print(fct_train_std_spatial.isel(lead_time = 0 ).msl.values)
        # 
        # =============================================================================
        
        
           
        # In[16]:
        # ### create batches
        print('create batches for fold {}'.format(fold_no))
        bs=32
        
        dg_train = DataGenerator(fct_train, verif_train,
                                 lead_input=lead_input,lead_output = lead_output,  
                                 basis = basis, clim_probs = clim_probs,
                                 batch_size=bs, load=True)
        
        #dg_train[0][1].shape
        #dg_train[-1][0][2].shape
        
        dg_valid = DataGenerator(fct_valid, verif_valid,
                                 lead_input=lead_input,lead_output = lead_output, 
                                 basis = basis, clim_probs = clim_probs, 
                                 batch_size=bs, load=True)
        print('finished creating batches for fold {}'.format(fold_no))
        # =============================================================================
        # dg_valid[0][1].shape
        # dg_valid[-1][0][0].shape
        # dg_valid[-1][0][1].shape
        # dg_valid[-1][0][2].shape
        # 
        # =============================================================================
        
        
        
        # In[20]:
        # ## CNN model
        
        
        # hyper-parameters for CNN
        dropout_rate = 0.4
        hidden_nodes = [10] #they tried different architectures
        
        
        # In[21]:
        
        # CNN: slightly adapted from Scheuerer et al 2020.
        
        inp_imgs = Input(shape=(dg_train[0][0][0].shape[1], dg_train[0][0][0].shape[2],dg_train[0][0][0].shape[3],)) #fcts4121,240,
        inp_basis = Input(shape=(n_xy,n_basis)) #basis
        inp_cl = Input(shape=(n_xy,n_bins,)) #climatology
        
        c = Conv2D(4, (3,3), activation='elu')(inp_imgs)
        c = MaxPooling2D((2,2))(c)
        #c = Conv2D(8, (3,3), activation='elu')(c)
        #c = MaxPooling2D((2,2))(c)
        x = Flatten()(c)
        for h in hidden_nodes: 
            x = Dropout(dropout_rate)(x)
            x = Dense(h, activation='elu')(x)
        x = Dense(n_bins*n_basis, activation='elu')(x)
        x = Reshape((n_bins,n_basis))(x)
        z = Dot(axes=2)([inp_basis, x])     # Tensor product with basis functions
        #z = Add()([z, inp_cl])              # Add (log) probability anomalies to log climatological probabilities 
        out = Activation('softmax')(z)
        
        
        cnn = Model(inputs=[inp_imgs, inp_basis, inp_cl], outputs=out)
        #cnn.summary()
        
        
        cnn.compile(loss= 'categorical_crossentropy', metrics=['accuracy'], optimizer=keras.optimizers.Adam(1e-4))#'adam')#my_loss_rps
        #keras.losses.CategoricalCrossentropy(label_smoothing = 0.0)
        
        
        
        # In[22]:
        
        #print('start fitting model in fold '.format(fold_no))
        ep  = 5
        hist =cnn.fit(dg_train,  
                         epochs=ep, shuffle = False,
                         validation_data = dg_valid)
        
        
        # In[23]:
         
        # ##save model performance
        scores = cnn.evaluate(dg_valid, verbose = 0)
        
        acc_per_fold.append(scores[1])
        loss_per_fold.append(scores[0])
        
        preds_single = single_prediction(cnn, 
                                     [fct_valid.sel(lead_time = lead_input).fillna(0.).to_array().transpose('forecast_time', ...,'variable').values, 
                                       np.repeat(basis[np.newaxis,:,:],len(valid_indeces),axis=0),
                                     np.repeat(clim_probs[np.newaxis,:,:],len(valid_indeces),axis=0)],
                                      lead_input, lons, lats, fct_valid, verif_train, lead_output)    
        preds_masked = preds_single.where(mask[v].sel(lead_time = lead_output).notnull())
    
        skill_year1.append(skill_by_year_single(preds_single, 
                         obs_2000_2019_terciled.isel(forecast_time=valid_indeces)[v].sel(lead_time = lead_output),
                         v).RPSS.values[0]#to_dict()
                     )
        skill_year2.append(skill_by_year_single(preds_single, 
                         obs_2000_2019_terciled.isel(forecast_time=valid_indeces)[v].sel(lead_time = lead_output),
                         v).RPSS.values[1]#to_dict()
                     )
        
        setup_params.append([fold_no, output_lat, output_lon, basis_rad, n_basis, standardization, rmcycle, rmterciles, bs, ep, list(fct_train.keys())])
        #param_rmcycle.append(rmcycle)
        #param_standardization.append(standardization)
        percent_min.append(preds_masked.min().values)
        percent_max.append(preds_masked.max().values)
        history.append(hist.history)
          

        # In[24]:
        
        #another 5 epochs
        ep  = 5
        hist =cnn.fit(dg_train,  
                         epochs=ep, shuffle = False,
                         validation_data = dg_valid)
        
        
        # In[25]:
         
        # ##save model performance
        scores = cnn.evaluate(dg_valid, verbose = 0)
        
        acc_per_fold.append(scores[1])
        loss_per_fold.append(scores[0])
        
        preds_single = single_prediction(cnn, 
                                     [fct_valid.sel(lead_time = lead_input).fillna(0.).to_array().transpose('forecast_time', ...,'variable').values, 
                                       np.repeat(basis[np.newaxis,:,:],len(valid_indeces),axis=0),
                                     np.repeat(clim_probs[np.newaxis,:,:],len(valid_indeces),axis=0)],
                                      lead_input, lons, lats, fct_valid, verif_train, lead_output)    
        preds_masked = preds_single.where(mask[v].sel(lead_time = lead_output).notnull())
    
        skill_year1.append(skill_by_year_single(preds_single, 
                         obs_2000_2019_terciled.isel(forecast_time=valid_indeces)[v].sel(lead_time = lead_output),
                         v).RPSS.values[0]#to_dict()
                     )
        skill_year2.append(skill_by_year_single(preds_single, 
                         obs_2000_2019_terciled.isel(forecast_time=valid_indeces)[v].sel(lead_time = lead_output),
                         v).RPSS.values[1]#to_dict()
                     )
        
        setup_params.append([fold_no, output_lat, output_lon, basis_rad, n_basis, standardization, rmcycle, rmterciles, bs, ep*2, list(fct_train.keys())])
        #param_rmcycle.append(rmcycle)
        #param_standardization.append(standardization)
        percent_min.append(preds_masked.min().values)
        percent_max.append(preds_masked.max().values)
        history.append(hist.history)
          
# In[30]: 

#results = pd.concat([param_standardization, param_rmcycle, acc_per_fold,loss_per_fold,skill], ignore_index=True)

setup = pd.DataFrame(setup_params, columns = ['fold_no','latitudes', 'longitudes', 'radius_basis_func', 'number_basis_func','standardization', 'rm_annual_cycle', 'rm_tercile_edges', 'batch_size', 'epochs', 'features'])
#setup_params, #fold_no, input_lat, output_lat, standardization, rmcycle, rmterciles, bs, ep, basis_rad, fct_train.keys()

results_ = pd.DataFrame(np.column_stack([acc_per_fold,loss_per_fold,skill_year1, skill_year2, percent_min, percent_max, history]), #
                               columns=['accuracy','loss','RPSS_year1', 'RPSS_year2', 'min_percentage', 'max_percentage', 'history']) 

results = pd.concat([setup, results_], axis = 1)        
# In[31]: 
    
results.to_csv('results_t2m_34_t2m.csv', sep =';', index = False)   

#########for loop ends here
print('finished kfold cross-validation')    

# In[30]:
# don't do this  in cross-val since it will slow down everything

# #### prediction from CNN
# =============================================================================
# pred_time = fct_valid.forecast_time#.isel(forecast_time = slice (0,10))
# 
# preds_single = single_prediction(cnn, 
#                                  [fct_valid.sel(forecast_time = pred_time).sel(lead_time = lead_input).fillna(0.).to_array().transpose('forecast_time', ...,'variable').values,
#                                   #fct_valid.sel(lead_time = lead_input).transpose('forecast_time',...).fillna(0.).values,  
#                                    np.repeat(basis[np.newaxis,:,:],len(pred_time),axis=0),
#                                  np.repeat(clim_probs[np.newaxis,:,:],len(pred_time),axis=0)],
#                                   lead_input, lons, lats, fct_valid.sel(forecast_time = pred_time), verif_train, lead_output)
#         
# 
# preds_masked = preds_single.where(mask[v].sel(lead_time = lead_output).notnull())
# =============================================================================

# In[31]:
    
#preds_masked.isel(forecast_time = 0).plot(col = 'category')#, vmin = 0, vmax = 1)




# In[32]:
# #### ground truth

#verif_valid.sel(lead_time = lead_output).isel(forecast_time = 0).plot(col = 'category')


# In[34]:
# #### tercile probs of the raw ensemble

# =============================================================================
# test = hind_2000_2019.isel(forecast_time=valid_indeces)[v]
# 
# 
# tercile_edges = load_data(data = 'obs_tercile_edges_2000-2019', aggregation = 'biweekly', path = path_data)
# test_raw = make_probabilistic(test, tercile_edges)
# 
# test_raw.isel(lead_time = 0).isel(forecast_time = 0)['t2m'].where(mask[v].sel(lead_time = lead_output).notnull()).plot(col = 'category')
# 
# =============================================================================

# In[35]:
# ## Compute RPSS

#computes RPSS wrt climatology (1/3 for each category. So, negative RPSS are worse than climatology...


# =============================================================================
# skill_prediction = skill_by_year_single(preds_single, 
#                      obs_2000_2019_terciled.isel(forecast_time=valid_indeces)[v].sel(lead_time = lead_output),
#                      v)
# 
# print(skill_prediction)
# =============================================================================
# RPSS in the order of -0.015 (for r_basis = 14)
# if you choose a larger radius for the basis functions, we might come closer to climatology and hence achieve a better RPSS.
# fluctuations in predictions are large.


# In[36]:


# =============================================================================
# skill_raw = skill_by_year_single(test_raw[v].sel(latitude = output_lat, longitude = output_lon).sel(lead_time = lead_output),
#                      obs_2000_2019_terciled.isel(forecast_time=valid_indeces)[v].sel(lead_time = lead_output),
#                      v)
# 
# print(skill_raw)
# =============================================================================
# RPSS in the order of -0.635


# #### The RPSS of this CNN approach is higher than for all ANNs. 
# This is probably because this approach predicts the smoothest fields and relaxes the most towards climatology.
# Actually, the CNN fields are less smooth than the fields of ANN terciled. 



# In[40]:

    
# # Reproducibility

# ## memory
# https://phoenixnap.com/kb/linux-commands-check-memory-usage
#get_ipython().system('free -g')

# ## CPU
#get_ipython().system('lscpu')

# ## software
#get_ipython().system('conda list')
