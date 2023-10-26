import data_exploration as de
import data_preprocessing as dp
import consecutive_subsets as cs
dataset_path = "your_path"
train_path = dataset_path+"/train"
test_path = dataset_path+"/test"
destination_path = "your_path"

############## new dataset configurations ################

de.check_data(dataset_path)
labels, tam_labels = de.get_labels(train_path)
de.plot_n_images(train_path,labels,3)

de.shape_labels(labels,train_path)

flatten_dataset, directory = de.get_flatten_dataset(train_path,labels)
new_labels = de.flatten_labels(labels,train_path)

#### adding label and file directory
dataset = dp.make_dataset_ps(flatten_dataset,new_labels,directory)

#################### PCA ##############################
pca_dataset = dp.pca_method(flatten_dataset)
pca_dataset = dp.make_dataset_ps(pca_dataset,new_labels,directory)
dp.plot_dr_figure(pca_dataset.iloc[:,:-2],new_labels)

####################### T-SNE  #####################
tsne_dataset = dp.tsne_method(flatten_dataset)
tsne_dataset = dp.make_dataset_ps(tsne_dataset,new_labels,directory)
dp.plot_dr_figure(tsne_dataset.iloc[:,:-2],new_labels)

####################### SVM  #####################
svm_dataset = dp.osvm_ps(dataset, "poly")
dp.make_new_dataset ("SVM",destination_path ,train_path, svm_dataset,labels)
####################### ISO  #####################
iso_dataset = dp.iso_ps(dataset, 0.5 )
dp.make_new_dataset ("ISO",destination_path ,train_path, iso_dataset,labels)
#################### PCA + SVM ########################
pca_svm_dataset = dp.pca_method(svm_dataset.iloc[:,:-2])
dp.plot_dr_figure(pca_svm_dataset,svm_dataset.iloc[:,-2])
#################### PCA + ISO ########################
pca_iso_dataset = dp.pca_method(iso_dataset.iloc[:,:-2])
dp.plot_dr_figure(pca_iso_dataset,iso_dataset.iloc[:,-2])

####################### TSNE - SVM  #####################
tsne_svm_dataset = dp.tsne_method(svm_dataset.iloc[:,:-2])
dp.plot_dr_figure(tsne_svm_dataset,svm_dataset.iloc[:,-2])
####################### TSNE - ISO  #####################
tsne_iso_dataset = dp.tsne_method(iso_dataset.iloc[:,:-2])
dp.plot_dr_figure(tsne_iso_dataset,iso_dataset.iloc[:,-2])
print(dataset.iloc[:,:-2])
######################### PS  #######################

ps_data = dp.prototype_selection (dataset, labels, 0.6)
print(len(ps_data))
tsne_ps_dataset = dp.tsne_method(ps_data.iloc[:,:-2])
dp.plot_dr_figure(tsne_ps_dataset,ps_data.iloc[:,-2])
dp.make_new_dataset ("PS", destination_path,
                     train_path, ps_data,labels)


#get consecutive subsets:
samples=[10,25,50]

#create folders
for i in samples:
    cs.create_folders(dataset_path + "/", train_path,labels,i)
#first subset
cs.small_batch(dataset_path + "/", train_path,labels,10)
#get the rest subsets
for i in range (len(samples)-1):
    cs.get_images(dataset_path + "/",train_path,labels,samples[i],samples[i+1])

