""" 
week 2 forecasts as additional predictors for week 3-4?
"""

import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Reshape, Dot, Add, Activation

import numpy as np
import pandas as pd

import xarray as xr
xr.set_options(display_style='text')

from scripts import add_year_week_coords
from helper_ml_data import load_data, get_basis, rm_annualcycle, rm_tercile_edges, DataGenerator1, single_prediction, skill_by_year_single

import warnings
warnings.simplefilter("ignore")

# In[2]:
### set seed
np.random.seed(42)

# In[5]:
# ### define some vars
    
v='t2m'

# 2 bi-weekly `lead_time`: week 3-4
lead_input = 0
lead_output = 0
#%%

#lists to store results of optimization experiments
setup_params = []
acc_per_fold = [] 
loss_per_fold = []
skill_year1 = []
skill_year2 = []
percent_min = []
percent_max = []
history = []
    
#%%

path_data = 'local_n'#'server'#['../../../../Data/s2s_ai/data', '../../../../Data/s2s_ai']

cases = ['w34', 'w2', 'w234']
for j in cases:
    print(j)

    if j == 'w34':
        # ## Hindcast
        hind_2000_2019_raw = load_data(data = 'hind_2000-2019', aggregation = 'biweekly', path = path_data,var_list = ['t2m']).isel(lead_time = lead_output)
        if 'sm20' in hind_2000_2019_raw.keys():
            hind_2000_2019_raw = hind_2000_2019_raw.isel(depth_below_and_layer = 0 ).reset_coords('depth_below_and_layer', drop=True)
        elif 'msl' in hind_2000_2019_raw.keys():
            hind_2000_2019_raw = hind_2000_2019_raw.isel(meanSea = 0).reset_coords('meanSea', drop = True)
        hind_2000_2019_raw = hind_2000_2019_raw[['t2m']]
    
    elif j =='w2':
                # ## Hindcast
        hind_2000_2019_raw = load_data(data = 'hind_2000-2019', aggregation = 'weekly', path = path_data,var_list = ['t2m']).isel(lead_time = lead_input)
        if 'sm20' in hind_2000_2019_raw.keys():
            hind_2000_2019_raw = hind_2000_2019_raw.isel(depth_below_and_layer = 0 ).reset_coords('depth_below_and_layer', drop=True)
        elif 'msl' in hind_2000_2019_raw.keys():
            hind_2000_2019_raw = hind_2000_2019_raw.isel(meanSea = 0).reset_coords('meanSea', drop = True)
        hind_2000_2019_raw = [hind_2000_2019_raw[n].rename(f'{n}_2') for n in list(hind_2000_2019_raw.keys())]
        hind_2000_2019_raw = xr.merge(hind_2000_2019_raw)
        hind_2000_2019_raw = hind_2000_2019_raw[['t2m_2']]
    
    elif j == 'w234':  
        hind_2000_2019_raw_34 = load_data(data = 'hind_2000-2019', aggregation = 'biweekly', path = path_data,var_list = ['t2m']).isel(lead_time = lead_output)
        if 'sm20' in hind_2000_2019_raw_34.keys():
            hind_2000_2019_raw_34 = hind_2000_2019_raw_34.isel(depth_below_and_layer = 0 ).reset_coords('depth_below_and_layer', drop=True)
        elif 'msl' in hind_2000_2019_raw_34.keys():
            hind_2000_2019_raw_34 = hind_2000_2019_raw_34.isel(meanSea = 0).reset_coords('meanSea', drop = True)
    
    
        hind_2000_2019_raw_2 = load_data(data = 'hind_2000-2019', aggregation = 'weekly', path = path_data,var_list = ['t2m']).isel(lead_time = lead_input)
        if 'sm20' in hind_2000_2019_raw_2.keys():
            hind_2000_2019_raw_2 = hind_2000_2019_raw_2.isel(depth_below_and_layer = 0 ).reset_coords('depth_below_and_layer', drop=True)
        elif 'msl' in hind_2000_2019_raw_2.keys():
            hind_2000_2019_raw_2 = hind_2000_2019_raw_2.isel(meanSea = 0).reset_coords('meanSea', drop = True)
        
        hind_2000_2019_raw_2 = [hind_2000_2019_raw_2[n].rename(f'{n}_2') for n in list(hind_2000_2019_raw_2.keys())]
        hind_2000_2019_raw_2 = xr.merge(hind_2000_2019_raw_2)#, overwrite = True)
        
        hind_2000_2019_raw = xr.merge([hind_2000_2019_raw_2,hind_2000_2019_raw_34], compat = 'override')
        hind_2000_2019_raw = hind_2000_2019_raw[['t2m','t2m_2']]
    # ## Observations corresponding to hindcasts
    obs_2000_2019_raw = load_data(data = 'obs_2000-2019', aggregation = 'biweekly', path = path_data).isel(lead_time = lead_output)
    # terciled
    obs_2000_2019_terciled_raw = load_data(data = 'obs_terciled_2000-2019', aggregation = 'biweekly', path = path_data).isel(lead_time = lead_output)
    
    hind_2000_2019_raw.assign_coords(lead_time = obs_2000_2019_raw.lead_time)##overwrite lead_time... not ideal but ow strange issues...
# In[4]:

    input_lats = [slice(81,34)]
    input_lons = [slice(49,97)]
    
    output_lats = [slice(63,52)]#4, 8, 16, 32
    output_lons = [slice(67,78)]
    
    i = 0
    input_lon = input_lons[i]
    input_lat = input_lats[i]
    
    output_lon = output_lons[i]
    output_lat = output_lats[i]

    rad = [8]
    basis_rad = rad[i]
        
    smoothing = 0.4 #label smoothing in categorical cross entropy

    hind_2000_2019 = hind_2000_2019_raw.sel(longitude = input_lon, latitude = input_lat)
    obs_2000_2019 = obs_2000_2019_raw.sel(longitude = output_lon, latitude = output_lat)
    obs_2000_2019_terciled = obs_2000_2019_terciled_raw.sel(longitude = output_lon, latitude = output_lat)

    # In[6]:
    
    #mask: same missing values at all forecast_times, only used for label data
    mask = xr.where(obs_2000_2019.notnull(),1,np.nan).mean('forecast_time', skipna = False)
    
    # In[7]:
    # #### compute basis
    
    basis, lats, lons, n_xy, n_basis = get_basis(obs_2000_2019_terciled[v], basis_rad)
    ##the smaller you choose the radius of the basis functions, the more memory needs to be allocated
    
    # In[8]:
    # ## climatology
    
    n_bins = 3 #we have to predict probs for 3 bins
    clim_probs = np.log(1/3)*np.ones((n_xy,n_bins))
    
    # In[9]:
    ###crossval 
    
    standardization = True
    rmcycle = True
    rmterciles = True

    # K-fold Cross Validation model evaluation
    for fold_no in range(0,10):
        print('fold {}'.format(fold_no))
    
        valid_indeces = range(fold_no*2*53, fold_no*2*53 + 53*2)
        train_indeces = [i for i in range(53*20) if i not in valid_indeces]

        # In[10]:
        # ### create datasets
        
        #train
        fct_train = hind_2000_2019.isel(forecast_time=train_indeces)
          
        verif_train = obs_2000_2019_terciled.isel(forecast_time=train_indeces)[v]
        verif_train = verif_train.where(mask[v].notnull())
          
        #validation
        fct_valid = hind_2000_2019.isel(forecast_time=valid_indeces)
        
        verif_valid = obs_2000_2019_terciled.isel(forecast_time=valid_indeces)[v]
        verif_valid = verif_valid.where(mask[v].notnull())
        
        # In[11]:
        # ### input features
        
        #use ensemble mean as input
        fct_train = fct_train.mean('realization')
        fct_valid = fct_valid.mean('realization')
        
        # In[12]:
        # ### Remove annual cycle
        
        ### dont remove annual cycle from land-sea mask!
        # ### land sea mask varies over time and ensemble members!
        if rmcycle == True:
            var_list = [i for i in fct_train.keys() if i not in ['lsm',v,f'{v}_2']]
            
            if len(var_list) == 1:
                fct_train[var_list] = rm_annualcycle(fct_train[var_list], fct_train[var_list])[var_list[0]]
                fct_valid[var_list] = rm_annualcycle(fct_valid[var_list], fct_valid[var_list])[var_list[0]]
            elif len(var_list) >1:
                fct_train[var_list] = rm_annualcycle(fct_train[var_list], fct_train[var_list])
                fct_valid[var_list] = rm_annualcycle(fct_valid[var_list], fct_valid[var_list])
         
        
        def rm_tercile_edges(ds, ds_train):#tercile_edges):
            #remove annual cycle for each location 
            
            ds = add_year_week_coords(ds)
            ds_train = add_year_week_coords(ds_train)
            
            tercile_edges = ds_train.chunk({'forecast_time':-1,'longitude':'auto'}).groupby('week').quantile(q=[1./3.,2./3.], dim='forecast_time').rename({'quantile':'category_edge'}).astype('float32')#groupby('week').mean(['forecast_time'])
        
            ds_stand = tercile_edges - ds
        
            ds_stand = ds_stand.sel({'week' : ds.coords['week']})
            ds_stand = ds_stand.drop(['week','year'])
            #ds_stand
            return ds_stand
  
        if rmterciles == True: 
            vars_ = [i for i in fct_train.keys() if i in [v,f'{v}_2']]
            
            #train
            deb = rm_tercile_edges(fct_train[vars_], fct_train[vars_])
            debs = []
            for var in list(deb.keys()):
                debs.append(deb[var].assign_coords( category_edge = ['lower_{}'.format(var),'upper_{}'.format(var)]).to_dataset(dim ='category_edge'))
            deb_train = xr.merge(debs)
            
            #valid
            deb = rm_tercile_edges(fct_valid[vars_], fct_train[vars_])
            debs = []
            for var in list(deb.keys()):
                debs.append(deb[var].assign_coords( category_edge = ['lower_{}'.format(var),'upper_{}'.format(var)]).to_dataset(dim ='category_edge'))
            deb_valid = xr.merge(debs)
            
            fct_train = xr.merge([fct_train, deb_train])
            fct_train = fct_train.drop_vars(vars_)
            
            fct_valid = xr.merge([fct_valid, deb_valid])
            fct_valid = fct_valid.drop_vars(vars_)
        
        # In[13]:
        #standardization:

        if standardization == True:
            #normalisation constant for each field
            fct_train_std_spatial = fct_train.std(('latitude','longitude')).mean(('forecast_time'))
            #mean over 'realization', already computed : this is the spatial standard deviation of the ensemble mean
            
            #do the normalization
            fct_train = fct_train / fct_train_std_spatial
            fct_valid = fct_valid / fct_train_std_spatial#use train set for standardization
           
        # In[16]:
        # ### create batches
        print('create batches for fold {}'.format(fold_no))
        bs=32
        
        dg_train = DataGenerator1(fct_train, verif_train,  
                                 basis = basis, clim_probs = clim_probs,
                                 batch_size=bs, load=True)

        dg_valid = DataGenerator1(fct_valid, verif_valid, 
                                 basis = basis, clim_probs = clim_probs, 
                                 batch_size=bs, load=True)
        print('finished creating batches for fold {}'.format(fold_no))

        # In[20]:
        # ## CNN model
        
        # hyper-parameters for CNN
        dropout_rate = 0.4
        hidden_nodes = [10] #they tried different architectures
        
        # In[21]:
        
        # CNN: slightly adapted from Scheuerer et al 2020.
        
        inp_imgs = Input(shape=(dg_train[0][0][0].shape[1], dg_train[0][0][0].shape[2],dg_train[0][0][0].shape[3],)) #fcts
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
        
        cnn.compile(loss= keras.losses.CategoricalCrossentropy(label_smoothing = smoothing), metrics=['accuracy'], optimizer=keras.optimizers.Adam(1e-4))
        
        # In[22]:

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
                                     [fct_valid.fillna(0.).to_array().transpose('forecast_time', ...,'variable').values,
                                       np.repeat(basis[np.newaxis,:,:],len(valid_indeces),axis=0),
                                     np.repeat(clim_probs[np.newaxis,:,:],len(valid_indeces),axis=0)],
                                      lead_input, lons, lats, fct_valid, verif_train, lead_output)    
        preds_masked = preds_single.where(mask[v].notnull())
    
        skill_year1.append(skill_by_year_single(preds_single, 
                         obs_2000_2019_terciled.isel(forecast_time=valid_indeces)[v],
                         v).RPSS.values[0]
                     )
        skill_year2.append(skill_by_year_single(preds_single, 
                         obs_2000_2019_terciled.isel(forecast_time=valid_indeces)[v],
                         v).RPSS.values[1]
                     )
        
        setup_params.append([fold_no, output_lat, output_lon, input_lat, input_lon, basis_rad, smoothing, n_basis, standardization, rmcycle, rmterciles, bs, ep, list(fct_train.keys())])
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
                                     [fct_valid.fillna(0.).to_array().transpose('forecast_time', ...,'variable').values,
                                       np.repeat(basis[np.newaxis,:,:],len(valid_indeces),axis=0),
                                     np.repeat(clim_probs[np.newaxis,:,:],len(valid_indeces),axis=0)],
                                      lead_input, lons, lats, fct_valid, verif_train, lead_output)    
        preds_masked = preds_single.where(mask[v].notnull())
    
        skill_year1.append(skill_by_year_single(preds_single, 
                         obs_2000_2019_terciled.isel(forecast_time=valid_indeces)[v],
                         v).RPSS.values[0]
                     )
        skill_year2.append(skill_by_year_single(preds_single, 
                         obs_2000_2019_terciled.isel(forecast_time=valid_indeces)[v],
                         v).RPSS.values[1]
                     )
        
        setup_params.append([fold_no, output_lat, output_lon, input_lat, input_lon, basis_rad, smoothing, n_basis, standardization, rmcycle, rmterciles, bs, ep*2, list(fct_train.keys())])
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
                                     [fct_valid.fillna(0.).to_array().transpose('forecast_time', ...,'variable').values,
                                       np.repeat(basis[np.newaxis,:,:],len(valid_indeces),axis=0),
                                     np.repeat(clim_probs[np.newaxis,:,:],len(valid_indeces),axis=0)],
                                      lead_input, lons, lats, fct_valid, verif_train, lead_output)    
        preds_masked = preds_single.where(mask[v].notnull())
    
        skill_year1.append(skill_by_year_single(preds_single, 
                         obs_2000_2019_terciled.isel(forecast_time=valid_indeces)[v],
                         v).RPSS.values[0]
                     )
        skill_year2.append(skill_by_year_single(preds_single, 
                         obs_2000_2019_terciled.isel(forecast_time=valid_indeces)[v],
                         v).RPSS.values[1]
                     )
        
        setup_params.append([fold_no, output_lat, output_lon, input_lat, input_lon, basis_rad, smoothing, n_basis, standardization, rmcycle, rmterciles, bs, ep*3, list(fct_train.keys())])
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
                                     [fct_valid.fillna(0.).to_array().transpose('forecast_time', ...,'variable').values,
                                       np.repeat(basis[np.newaxis,:,:],len(valid_indeces),axis=0),
                                     np.repeat(clim_probs[np.newaxis,:,:],len(valid_indeces),axis=0)],
                                      lead_input, lons, lats, fct_valid, verif_train, lead_output)    
        preds_masked = preds_single.where(mask[v].notnull())
    
        skill_year1.append(skill_by_year_single(preds_single, 
                         obs_2000_2019_terciled.isel(forecast_time=valid_indeces)[v],
                         v).RPSS.values[0]
                     )
        skill_year2.append(skill_by_year_single(preds_single, 
                         obs_2000_2019_terciled.isel(forecast_time=valid_indeces)[v],
                         v).RPSS.values[1]
                     )
        
        setup_params.append([fold_no, output_lat, output_lon, input_lat, input_lon, basis_rad, smoothing, n_basis, standardization, rmcycle, rmterciles, bs, ep*4, list(fct_train.keys())])
        percent_min.append(preds_masked.min().values)
        percent_max.append(preds_masked.max().values)
        history.append(hist.history)
# In[30]: 
#########for loop ends here

#gather results
setup = pd.DataFrame(setup_params, columns = ['fold_no','output_lats', 'output_lons', 'input_lats', 'input_lons', 
                                              'radius_basis_func', 'label_smoothing', 'number_basis_func','standardization', 
                                              'rm_annual_cycle', 'rm_tercile_edges', 'batch_size', 'epochs', 'features'])

results_ = pd.DataFrame(np.column_stack([acc_per_fold,loss_per_fold,skill_year1, skill_year2, percent_min, percent_max, history]), #
                               columns=['accuracy','loss','RPSS_year1', 'RPSS_year2', 'min_percentage', 'max_percentage', 'history']) 

results = pd.concat([setup, results_], axis = 1)        
# In[31]: 
    
#results.to_csv('E4_results_t2m_34_t2m.csv', sep =';', index = False)   