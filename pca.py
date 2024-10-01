
import pandas as pd
import pandas
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy as sp


def read_med(file, finfo, var_cols, col_n='C:'):
    """ Function to read-write MED raw data to csv
    :param file: String. The name of the file: path/to/file_name.
    :param finfo: A list with the subject, session number in a list
    :param path_tidy: String. Where to store processed data frame.
    :param var_cols: The number of columns in PRINTCOLUMNS.
    :param col_n: String. Column to extract information. By default 'C:'.
    :return: dataframe
    """
    # names parameter takes into account a data frame with ncols columns
    ncols = var_cols + 2
    df = pd.read_csv(file, delim_whitespace=True, header=None,
                      skiprows=3, names=['x' + str(i) for i in range(1, ncols)])
    subj = df.x2[2]
    subjf = finfo[0]
    a = np.where(df.x1 == "0:")[0]
    col_pos = np.where(df.x1 == col_n)[0]
    # Check whether subj name in fname and inside file is the same,
    # otherwise break and check fname for errors
    if subj != subjf or len(col_pos) != 1:
        print(f"Subject name in file {finfo} is wrong; or")
        print(f'{col_n} is not unique, possible error in {finfo}. Dup file?')
        stop = True
    else:
        stop = False
    while not stop:
        if sum(a == col_pos + 1)==0:
            vC=pd.DataFrame(columns=['x'])
            return vC
        else:
            col_idx = int(np.where(a == col_pos + 1)[0])
            start = a[col_idx]
            if a[col_idx] == a[-1]:
                end = len(df.index)
            else:
                end = a[col_idx + 1] - 2
            vC = df[start:(end + 1)].reset_index(drop=True)
            vC = vC.drop(columns='x1').stack().reset_index(drop=True)  # dropna default True
            #FutureWarning: In a future version of pandas all arguments of DataFrame.drop 
            #except forres the argument 'labels' will be keyword-only
            vC = np.asarray(vC, dtype='float64')
            vC = pd.DataFrame({'vC': vC.astype(str)})
            reach = True
            if reach:
                return vC

def load_formattedMedPC_df(home_dir, mice):
    Columns=['Mouse', 'Date', 'Event', 'Timestamp']
    events=[
        'ActiveLever', 
        'Lick',
        'No Response',
        'Cue'
        ] 
    arrays=[ 
        'J:', 
        'O:',
        'P:',
        'L:'
        ]
    Med_log=pd.DataFrame(columns=Columns)
    
    for mouse in mice:
        directory=os.path.join(home_dir, mouse)
        files = [f for f in os.listdir(directory) 
                  if (os.path.isfile(os.path.join(directory, f)) and f[0]!='.')]
        files.sort()
        for i,f in enumerate(files):
            print(f)
            date=f[:16]
            
            for event,col_n in zip(events, arrays):
                Timestamps=read_med(os.path.join(directory, f),[f[-9:-4], f[:10]], var_cols=5,col_n=col_n) 
                #Timestamps is a dataframe
                if len(Timestamps)!=0:
                    Timestamps.columns=['Timestamp']
                    Timestamps.at[:,'Mouse']=mouse
                    Timestamps.at[:,'Date']=date
                    Timestamps.at[:,'Event']=event
                    Med_log=pd.concat([Med_log,Timestamps],ignore_index=True) 
                elif len(Timestamps)== 0:
                    Timestamps.columns=['Timestamp']
                    Timestamps.at[:,'Timestamp']= [0]
                    Timestamps.at[:,'Mouse']=mouse
                    Timestamps.at[:,'Date']=date
                    Timestamps.at[:,'Event']=event
                    Med_log=pd.concat([Med_log,Timestamps],ignore_index=True) 
    #Format 'Timestamp' from str to float
    Med_log['Timestamp'] = Med_log['Timestamp'].astype(float)
    # Add a column called 'Session'
    for mouse in mice:
        mouse_log=Med_log.loc[np.equal(Med_log['Mouse'], mouse)]
        for i,day in enumerate(np.unique(mouse_log['Date'])):
            day_log=mouse_log.loc[np.equal(mouse_log['Date'], day)]
            Med_log.at[day_log.index, 'Session']=i
    return Med_log

home_dir = '/Users/kristineyoon/Library/CloudStorage/OneDrive-Vanderbilt/EtOHSA_SexDiff/PCA'
mice = ['SD1-1','SD1-2','SD1-4','SD1-5','SD2-1','SD2-2','SD2-3','SD2-4','SD2-5','SD3-1','SD3-2','SD3-4','SD3-5','SD4-1','SD4-2','SD4-3','SD4-4','SD4-5','SD5-1','SD5-2','SD6-1','SD6-2']
sip_access = 10
mpc_df = load_formattedMedPC_df(home_dir, mice)
mpc_df.to_csv('Med_log.csv', index = False)

org_df=pd.DataFrame()
lever_latency_dict = {}
by_session = {}
maxsession = int(max(mpc_df['Session'].values))
for j in range (0,int(max(mpc_df['Session'].values))+1):
    lickspermouse = []
    for mouse in mice:
        cue_time = []
        lick_time = []
        lever_time = []
        noresp_time = []
        
        for i in range(len(mpc_df)):
            if mpc_df.iloc[i,0] == mouse and mpc_df.iloc[i,4] == j and mpc_df.iloc[i,2] == 'Cue':
                cue_time.append(mpc_df.iloc[i,3])
            if mpc_df.iloc[i,0] == mouse and mpc_df.iloc[i,4] == j and mpc_df.iloc[i,2] == 'Lick':
                lick_time.append(mpc_df.iloc[i,3])
            if mpc_df.iloc[i,0] == mouse and mpc_df.iloc[i,4] == j and mpc_df.iloc[i,2] == 'ActiveLever':
                lever_time.append(mpc_df.iloc[i,3])
            
        print (str('Mouse: ' ) + str(mouse) + str(' Session: ') + str(j))
        #### First Magazine Session Training Lick ####
        if j == 0:
            org_df.at['MT Licks',mouse]=len(lick_time)
            
        else:
            #### Response ####
            if lever_time[0] == 0.0:
                org_df.at[f'ITI {j} Responses',mouse] = 0
            else:
                org_df.at[f'ITI {j} Responses',mouse]=len(lever_time)
            #### Latency to Press ####
            if len(cue_time) != 0:
                leverlatency = []
                for i in range (len(cue_time)):
                    for k in range (len(lever_time)):
                        if lever_time[k] - cue_time[i] <= 20 and lever_time[k] - cue_time[i] > 0:
                            leverlatency.append(lever_time[k] - cue_time[i])
                            
                if len(leverlatency) != 0:
                    avg_latency = sum(leverlatency)/len(leverlatency)
                else:
                    avg_latency = 0
            else:
                 avg_latency = 0
            org_df.at[f'ITI {j} Lever Latency',mouse]=avg_latency

            #### Latency to Lick ####
            if len(lever_time) != 0:
                licklatency = []
                for k in range (0,len(lever_time)):
                    for i in range (len(lick_time)):
                        if lick_time[i] - lever_time[k] <= sip_access and lick_time[i] - lever_time[k] > 0:
                            licklatency.append(lick_time[i]- lever_time[k])
                if len(licklatency) != 0:
                    avglicklatency = sum(licklatency)/len(licklatency)
                else:
                    avglicklatency = 0
            else:
                avglicklatency = 0
            org_df.at[f'ITI {j} Lick Latency',mouse]=avglicklatency  
        
        #### Interlick Interval ####
        
        interlickint_all = [lick_time[i + 1] - lick_time[i] for i in range(len(lick_time)-1)]
        interlickint_bytrial = [k for k in interlickint_all if k < 10]
        if len(interlickint_bytrial) > 0:
            interlickint_bytrial_avg = sum(interlickint_bytrial)/len(interlickint_bytrial)
        else:
            interlickint_bytrial_avg = 0
        org_df.at[f'Session {j} Interlick Interval',mouse]=interlickint_bytrial_avg
        
        #### Longest Lick Bout ####
        boutbreaks = np.array(interlickint_all) > 1
        boutindex = np.where(boutbreaks == True)
        boutindex = boutindex[0]
        boutlength = [0]
        if len(boutindex) > 0:
            boutall = []
            start = lick_time[0]
            for i in range(len(boutindex)):
                boutall.append([start,lick_time[boutindex[i]]])
                start = lick_time[boutindex[i]+1]
            
            for beg,end in boutall:
                boutlength.append(end-beg)
        org_df.at[f'Session {j} Longest Lick Bout Time',mouse] = max(boutlength)
        
        # boutcount = [0]
        # for beg,end in boutall:
        #     boutindicate = []
        #     for tstamp in lick_time:
        #         if tstamp >= beg and tstamp <= end:
        #             boutindicate.append(tstamp)
        #     boutcount.append(len(boutindicate))
       
        # org_df.at[f'Session {j} Longest Lick Bout #',mouse] = max(boutcount)
        
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import scipy.stats

org_zscore = scipy.stats.zscore(org_df, axis=1)
print(org_df)
matrix = np.array(org_zscore)
matrix = matrix.T

df =  pd.DataFrame(matrix, columns=['MT Licks', 'ITI 0 Interlick Interval',
       'ITI 0 Longest Lick Bout Time', 'ITI 1 Responses',
       'ITI 1 Lever Latency', 'ITI 1 Lick Latency',
       'ITI 1 Interlick Interval', 'ITI 1 Longest Lick Bout Time',
       'ITI 2 Responses', 'ITI 2 Lever Latency', 'ITI 2 Lick Latency',
       'ITI 2 Interlick Interval', 'ITI 2 Longest Lick Bout Time'])
pca = PCA(n_components=5)           # Initialize PCA with the number of components you want
pca.fit(df)                         # Fit PCA on your data
data_pc = pca.transform(df)         # Transform the original data into the principal component space
loadings = pca.components_          # Get the loadings for each principal component
loadings_pc1 = loadings[0]          # Access the loadings for PC1
loadings_pc2 = loadings[1] 

explained_variance_ratio = pca.explained_variance_ratio_
# Print the explained variance ratio for PCA1 and PCA2
print(f"Explained Variance Ratio for PCA1: {explained_variance_ratio[0]:.4f}")
print(f"Explained Variance Ratio for PCA2: {explained_variance_ratio[1]:.4f}")

# Create a DataFrame to show the contributions of each feature to PC1 (sorted)
contributions_pc1 = pd.DataFrame({'Feature': org_zscore.index, 'Contribution to PC1': loadings_pc1})
contributions_pc1 = contributions_pc1.reindex(contributions_pc1['Contribution to PC1'].abs().sort_values(ascending=False).index)
print(contributions_pc1)

# Standardize the data (important for k-means)
scaler = StandardScaler()
df_standardized = scaler.fit_transform(df)


# Fit PCA on the standardized data
pca_result = pca.fit_transform(df_standardized)

# Get the transformed data (with reduced dimensions)
df_pca = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2','PC3','PC4','PC5'])
plt.scatter(df_pca['PC1'], df_pca['PC2'], cmap='viridis', edgecolors='k')
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.show()
       

females = ['SD1-1','SD1-2','SD1-4','SD1-5', 'SD3-1','SD3-2','SD3-4','SD3-5','SD5-1','SD5-2' ]
males = ['SD2-1','SD2-2','SD2-3','SD2-4','SD2-5', 'SD4-1','SD4-2','SD4-3','SD4-4','SD4-5','SD6-1','SD6-2']
highdrinker = ['SD1-5', 'SD2-1', 'SD2-2', 'SD2-3' , 'SD2-4', 'SD2-5', 'SD3-2', 'SD3-5', 'SD4-1']


# BY SEX
plt.figure(figsize=(8, 6))
for i,mouse in enumerate(list(org_zscore)):
    if mouse in females:
        color = '#DE004A'
    else:
        color = '#108080'
    plt.scatter(df_pca['PC1'][i], df_pca['PC2'][i], color = color)
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.show()

# BY DRINKING PHENOTYPES
plt.figure(figsize=(8, 6))
for i,mouse in enumerate(list(org_zscore)):
    if mouse in highdrinker:
        color = 'red'
    else:
        color = 'green'
    plt.scatter(df_pca['PC1'][i], df_pca['PC2'][i], color = color)
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.show()

# text = list(org_zscore)
# for i in range (len(df)):
#     plt.annotate(text[i], (df_pca['PC1'][i], df_pca['PC2'][i] + 0.2)) 

# PC1 V. PC2 
plt.figure(figsize=(8, 6))
plt.scatter(df_pca['PC1'], df_pca['PC2'], cmap='viridis', edgecolors='k')
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')


# PC1 V. PC2 WITH LOADINGS WITH THE MALE V FEMALE
plt.figure(figsize=(16, 12), dpi=100)
plt.grid(True, alpha=0.2)
plt.xlabel('Principal Component 1 (25.37% Explained Variance)', size = 20)
plt.ylabel('Principal Component 2 (20.16% Explained Variance)', size = 20)
# Add arrows for feature loadings
for i, feature in enumerate(df.columns):
    if i in [3,4,8,7,9,12]:
        plt.arrow(0, 0, loadings_pc1[i]*4.5, loadings_pc2[i]*4.5, alpha=0.3, width=0.02, head_width=0.06)
text = list(df.columns)
for i in range (len(df)):
    if i in [3,4,8,12]:
        plt.annotate(text[i], (loadings_pc1[i]*4.5 + 0.1, loadings_pc2[i]*4.5 + 0.05), size = 16) 
    elif i in [7]:
        plt.annotate(text[i], (loadings_pc1[i]*4.5 + 0.2, loadings_pc2[i]*4.5 - 0.05), size = 16) 
    elif i in [3,8]:
        plt.annotate(text[i], (loadings_pc1[i]*4.5 + 0.2, loadings_pc2[i]*4.5 + 0.05), size = 16) 
    elif i in [9]:
        plt.annotate(text[i], (loadings_pc1[i]*4.5 -  0.2, loadings_pc2[i]*4.5 - 0.3), size = 16)
# BY SEX
for i,mouse in enumerate(list(org_zscore)):
    if mouse in females:
        color = '#DE004A'
        fem = plt.scatter(df_pca['PC1'][i], df_pca['PC2'][i], color = color, s=100.0)
    else:
        color = '#108080'
        mal = plt.scatter(df_pca['PC1'][i], df_pca['PC2'][i], color = color, s=100.0)  
plt.legend((fem,mal),('Female', 'Male'),loc='upper right', fontsize='16')
plt.savefig('/Users/kristineyoon/Documents/pca12_malefemale.pdf', transparent=True)
plt.show()


# PC1 V. PC2 WITH LOADINGS WITH THE HIGH LOW DRINKERS
plt.figure(figsize=(16, 12), dpi=100)
plt.grid(True, alpha=0.2)
plt.xlabel('Principal Component 1 (25.37% Explained Variance)', size = 20)
plt.ylabel('Principal Component 2 (20.16% Explained Variance)', size = 20)
# Add arrows for feature loadings
for i, feature in enumerate(df.columns):
    if i in [3,4,8,7,9,12]:
        plt.arrow(0, 0, loadings_pc1[i]*4.5, loadings_pc2[i]*4.5, alpha=0.3, width=0.02, head_width=0.06)
text = list(df.columns)
for i in range (len(df)):
    if i in [3,4,8,12]:
        plt.annotate(text[i], (loadings_pc1[i]*4.5 + 0.1, loadings_pc2[i]*4.5 + 0.05), size = 16) 
    elif i in [7]:
        plt.annotate(text[i], (loadings_pc1[i]*4.5 + 0.2, loadings_pc2[i]*4.5 - 0.05), size = 16) 
    elif i in [3,8]:
        plt.annotate(text[i], (loadings_pc1[i]*4.5 + 0.2, loadings_pc2[i]*4.5 + 0.05), size = 16) 
    elif i in [9]:
        plt.annotate(text[i], (loadings_pc1[i]*4.5 -  0.2, loadings_pc2[i]*4.5 - 0.3), size = 16)
for i,mouse in enumerate(list(org_zscore)):
    if mouse in highdrinker:
        color = '#E95C49'
        drinkhi = plt.scatter(df_pca['PC1'][i], df_pca['PC2'][i], color = color, s=100.0)
    else:
        color = '#A162A7'
        drinklo = plt.scatter(df_pca['PC1'][i], df_pca['PC2'][i], color = color, s=100.0)  
plt.legend((drinkhi,drinklo),('High Drinker', 'Low Drinker'),loc='upper right', fontsize='16')
plt.savefig('/Users/kristineyoon/Documents/pca12_highlow.pdf', transparent=True)
plt.show()

# PC1 AGAINST EACH ORIGINAL FEATURE 
plt.figure(figsize=(16, 12), dpi=100)
for i, feature in enumerate(df.columns):
    plt.scatter(data_pc[:, 0], df[feature], label=feature, s=50.0)
plt.xlabel('Principal Component 1 (PC1)', size = 20)
plt.ylabel('Original Feature Values', size = 20)
# plt.title('Scatter Plot of PC1 against Original Features', size = 20)
plt.legend(fontsize='13', ncol=4, columnspacing=0.1)
plt.savefig('/Users/kristineyoon/Documents/pca1_features.pdf', transparent=True)

# Add arrows for feature loadings
for i, feature in enumerate(df.columns):
    plt.arrow(0, 0, loadings_pc1[i]*2, loadings_pc1[i]*2, alpha=0.5, width=0.02)
plt.show()


# CLUSTERING
kmeans = KMeans(n_clusters=2, random_state=42)                                  # Apply k-means clustering
df_pca['Cluster'] = kmeans.fit_predict(df_pca[['PC1', 'PC2','PC3','PC4','PC5']])
centroids = scaler.inverse_transform(kmeans.cluster_centers_)                   # Get the centroids of each cluster in the original feature space
centroids_df = pd.DataFrame(data=centroids, columns=df.columns)                 # Create a DataFrame to display the centroids

plt.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['Cluster'], cmap='viridis', edgecolors='k')
plt.scatter(centroids_df['ITI 2 Responses'], centroids_df['ITI 2 Lever Latency'] ,marker='X', s=200, color='red', label='Centroids')
plt.title('PCA with K-Means Clustering and Centroids')
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.legend()
plt.show()

# Display the centroids in the original feature space
print("Centroids in Original Feature Space:")
print(centroids_df)



def pca_code(data): # from https://stackoverflow.com/questions/31909945/obtain-eigen-values-and-vectors-from-sklearn-pca
        #raw_implementation
        var_per=.98 # only keep eigenvalues/components up to 98% variance explained
        data-=np.mean(data, axis=0) # subtracting the men is necessary for PCA to track covariance. Otherwise the first PC will reflect the mean signal.
        cov_mat=np.cov(data, rowvar=False) #calculate covariance matrix
        evals, evecs = np.linalg.eigh(cov_mat) #perform eigenvalue decomposition on the covarince matrix (evals: list of weights of eigenvalues/components; evecs: eigevalue decomposition, should be #variables x #variables, each column is a component/eigenvalue, each row stores the weights of each variable on the PC)
        idx = np.argsort(evals)[::-1]# get index for PCs from biggest to smallest
        evecs = evecs[:,idx] # sort evecs matrix from biggest to smallest PC weight (rows still represent each variable, the columns are ordered from 1st to last PC)
        evals = evals[idx] #sort the PCs
        variance_retained=np.cumsum(evals)/np.sum(evals) # identify the cumulative variance explained for each PC
        index=np.argmax(variance_retained>=var_per) # get index of PCs that explain var_per
        #evecs = evecs[:,:index+1] # only keep the PCs up to the ones that cumulatively explain var_per
        reduced_data=np.dot(evecs.T, data.T).T #equivalent to pca.transform() (evecs would have been under the hood of pca.fit())
        #print("evals", evals)
        #print("_"*30)
        #print(evecs.T[1, :])
        #print("_"*30)
        #using scipy package
        #clf=PCA(var_per)
        #X_train=data
        #X_train=clf.fit_transform(X_train)
        #print(clf.explained_variance_)
        #print("_"*30)
        #print(clf.components_[1,:])
        #print("__"*30)
        return  evals, evecs, reduced_data,variance_retained


org_zscore = sp.stats.zscore(org_df, axis=1)
print(org_df)

matrix = np.array(org_zscore)
matrix = matrix.T
evals, evecs, reduced_data,variance_retained = pca_code(matrix)
plt.figure()
plt.scatter(reduced_data[:,0],reduced_data[:,1])


plt.figure()
for i,mouse in enumerate(list(org_zscore)):
    if mouse in females:
        color = 'pink'
    else:
        color = 'blue'
    plt.scatter(reduced_data[i,0],reduced_data[i,1], color = color)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()


plt.figure()
for i,mouse in enumerate(list(org_zscore)):
    if mouse in highdrinker:
        color = 'red'
    # elif mouse in lowdrinker:
    #     color = 'green'
    else:
        color = 'green'
    plt.scatter(reduced_data[i,0],reduced_data[i,1], c = color)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()

text = list(org_zscore)
for i in range (len(reduced_data)):
    plt.annotate(text[i], (reduced_data[i,0], reduced_data[i,1] + 0.2)) 

from sklearn.decomposition import PCA

# Initialize PCA with the number of components you want
pca = PCA(n_components=13)

# Fit PCA on your data
pca.fit(matrix)

# Get the loadings for each principal component
loadings = pca.components_

# Access the loadings for PC1
loadings_pc1 = loadings[0]

# Create a DataFrame to show the contributions of each feature to PC1
contributions_pc1 = pd.DataFrame({'Feature': org_zscore.index, 'Contribution to PC1': loadings_pc1})

# Sort by the absolute values to see the most influential features
contributions_pc1 = contributions_pc1.reindex(contributions_pc1['Contribution to PC1'].abs().sort_values(ascending=False).index)

print(contributions_pc1)


df =  pd.DataFrame(matrix, columns=org_zscore.index)

# Transform the original data into the principal component space
data_pc = pca.transform(df)

# Create a scatter plot of PC1 against each original feature
plt.figure(figsize=(8, 6))

for i, feature in enumerate(df.columns):
    if i in [3,4,7,6]:
        plt.scatter(data_pc[:, 0], df[feature], label=feature)

# Add labels and legend
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Original Feature Values')
plt.title('Scatter Plot of PC1 against Original Features')
plt.legend()

# Add arrows for feature loadings
for i, feature in enumerate(df.columns):
    if i in [3,4,7.6]:
        plt.arrow(0, 0, loadings_pc1[i]*3, loadings_pc1[i]*3, alpha=0.5, width=0.02)

plt.show()
