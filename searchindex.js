Search.setIndex({docnames:["Dataset","Dispatcher","FeatureSpec","Hooks","Initializers","Optimizer","Runner","index","tflibs","tflibs.datasets","tflibs.image","tflibs.model","tflibs.ops","tflibs.runner","tflibs.session","tflibs.training","tflibs.utils"],envversion:52,filenames:["Dataset.rst","Dispatcher.rst","FeatureSpec.rst","Hooks.rst","Initializers.rst","Optimizer.rst","Runner.rst","index.rst","tflibs.rst","tflibs.datasets.rst","tflibs.image.rst","tflibs.model.rst","tflibs.ops.rst","tflibs.runner.rst","tflibs.session.rst","tflibs.training.rst","tflibs.utils.rst"],objects:{"tflibs.datasets":{dataset:[0,0,0,"-"],feature_spec:[2,0,0,"-"]},"tflibs.datasets.dataset":{BaseDataset:[0,1,1,""]},"tflibs.datasets.dataset.BaseDataset":{add_arguments:[0,2,1,""],feature_specs:[0,3,1,""],read:[0,4,1,""],tfrecord_filename:[0,3,1,""],write:[0,4,1,""]},"tflibs.datasets.feature_spec":{FeatureSpec:[2,1,1,""],IDSpec:[2,1,1,""],ImageSpec:[2,1,1,""],LabelSpec:[2,1,1,""],MultiLabelSpec:[2,1,1,""]},"tflibs.datasets.feature_spec.FeatureSpec":{feature_proto:[2,4,1,""],feature_proto_spec:[2,3,1,""],parse:[2,4,1,""],shape:[2,3,1,""]},"tflibs.datasets.feature_spec.IDSpec":{create_with_string:[2,4,1,""],feature_proto_spec:[2,3,1,""],parse:[2,4,1,""]},"tflibs.datasets.feature_spec.ImageSpec":{create_with_contents:[2,4,1,""],create_with_path:[2,4,1,""],create_with_tensor:[2,4,1,""],feature_proto_spec:[2,3,1,""],parse:[2,4,1,""]},"tflibs.datasets.feature_spec.LabelSpec":{create_with_index:[2,4,1,""],feature_proto_spec:[2,3,1,""],from_class_names:[2,2,1,""],parse:[2,4,1,""]},"tflibs.datasets.feature_spec.MultiLabelSpec":{create_with_labels:[2,4,1,""],create_with_tensor:[2,4,1,""],feature_proto_spec:[2,3,1,""],from_class_names:[2,2,1,""],parse:[2,4,1,""]},"tflibs.image":{decode:[10,5,1,""],encode:[10,5,1,""]},"tflibs.model":{Model:[11,1,1,""]},"tflibs.model.Model":{add_eval_args:[11,2,1,""],add_model_args:[11,2,1,""],add_train_args:[11,2,1,""],model_fn:[11,6,1,""]},"tflibs.ops":{image:[12,0,0,"-"]},"tflibs.ops.image":{decode_image:[12,5,1,""],normalize:[12,5,1,""]},"tflibs.runner":{initializer:[4,0,0,"-"],runner:[6,0,0,"-"]},"tflibs.runner.initializer":{BaseInitializer:[4,1,1,""],DatasetInitializer:[4,1,1,""],ModelInitializer:[4,1,1,""],TrainInitializer:[4,1,1,""]},"tflibs.runner.initializer.BaseInitializer":{add_arguments:[4,4,1,""],handle:[4,4,1,""]},"tflibs.runner.initializer.DatasetInitializer":{add_arguments:[4,4,1,""],handle:[4,4,1,""]},"tflibs.runner.initializer.ModelInitializer":{add_arguments:[4,4,1,""],handle:[4,4,1,""]},"tflibs.runner.initializer.TrainInitializer":{add_arguments:[4,4,1,""],handle:[4,4,1,""]},"tflibs.runner.runner":{Runner:[6,1,1,""]},"tflibs.runner.runner.Runner":{argparser:[6,3,1,""],run:[6,4,1,""]},"tflibs.session":{generator:[14,5,1,""]},"tflibs.training":{dispatcher:[1,0,0,"-"],hooks:[3,0,0,"-"],optimizer:[5,0,0,"-"]},"tflibs.training.dispatcher":{Dispatcher:[1,1,1,""]},"tflibs.training.dispatcher.Dispatcher":{chief:[1,3,1,""],minimize:[1,4,1,""],models:[1,3,1,""]},"tflibs.training.hooks":{EvalSummaryHook:[3,1,1,""],EvaluationRunHook:[3,1,1,""],ImageSaverHook:[3,1,1,""]},"tflibs.training.hooks.EvalSummaryHook":{after_run:[3,4,1,""],before_run:[3,4,1,""],begin:[3,4,1,""]},"tflibs.training.hooks.EvaluationRunHook":{after_run:[3,4,1,""],before_run:[3,4,1,""]},"tflibs.training.hooks.ImageSaverHook":{after_run:[3,4,1,""],before_run:[3,4,1,""]},"tflibs.training.optimizer":{Optimizer:[5,1,1,""]},"tflibs.training.optimizer.Optimizer":{apply_gradients:[5,4,1,""],apply_tower_gradients:[5,4,1,""],compute_grad:[5,4,1,""],decay_learning_rate:[5,6,1,""],train_op:[5,4,1,""],var_list:[5,3,1,""]},"tflibs.utils":{data_structure:[16,0,0,"-"],decorators:[16,0,0,"-"],device_setter:[16,0,0,"-"],ioio:[16,0,0,"-"],logging:[16,0,0,"-"],pkg:[16,0,0,"-"],summary:[16,0,0,"-"]},"tflibs.utils.data_structure":{flatten_nested_dict:[16,5,1,""],map_dict:[16,5,1,""]},"tflibs.utils.decorators":{name_scope:[16,5,1,""],strip_dict_arg:[16,5,1,""]},"tflibs.utils.device_setter":{device_setter:[16,5,1,""]},"tflibs.utils.ioio":{download_file:[16,5,1,""]},"tflibs.utils.logging":{log_parse_args:[16,5,1,""]},"tflibs.utils.pkg":{import_module:[16,5,1,""],list_modules:[16,5,1,""]},"tflibs.utils.summary":{strip_illegal_summary_name:[16,5,1,""]},tflibs:{image:[10,0,0,"-"],model:[11,0,0,"-"],session:[14,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","classmethod","Python class method"],"3":["py","attribute","Python attribute"],"4":["py","method","Python method"],"5":["py","function","Python function"],"6":["py","staticmethod","Python static method"]},objtypes:{"0":"py:module","1":"py:class","2":"py:classmethod","3":"py:attribute","4":"py:method","5":"py:function","6":"py:staticmethod"},terms:{"class":[0,1,2,3,4,5,6,11],"default":6,"function":[0,6],"import":3,"int":0,"return":[0,6,10,16],"static":[5,11],"true":3,_mockobject:3,add:0,add_argu:[0,4],add_eval_arg:11,add_model_arg:11,add_train_arg:11,after_run:3,all:6,apply_gradi:5,apply_tower_gradi:5,argpars:[0,4,6,11],argument:0,argumentpars:0,arr:10,artifact:6,autodoc:3,base:[0,2,3,4],basedataset:0,baseiniti:4,before_run:3,begin:3,beta1:5,beta2:5,calcul:1,chief:1,class_nam:2,classmethod:[0,2,11],collect:0,compute_grad:5,condit:12,content:[2,7],create_with_cont:2,create_with_index:2,create_with_label:2,create_with_path:2,create_with_str:2,create_with_tensor:2,data:0,data_structur:16,dataset:[2,4,8],dataset_dir:0,dataset_pkg:4,datasetiniti:4,decay_it:5,decay_learning_r:5,decay_step:5,decod:10,decode_imag:12,decor:16,default_job_dir:6,defin:0,depend:1,depth:2,devic:[1,16],device_sett:16,dir:6,directori:[0,6],dispatch:[8,15],download_fil:16,encod:10,encoded_imag:12,estim:3,eval_step:3,evalsummaryhook:3,evaluationrunhook:3,exampl:0,ext:3,fals:[12,16],featur:[1,2],feature_proto:2,feature_proto_spec:2,feature_spec:[0,2],featurespec:[8,9],fetch:14,file:0,flatten_nested_dict:16,from_class_nam:2,gener:14,gpu:1,grads_and_var:5,handl:4,hook:[8,15],idspec:2,illeg:16,imag:[3,8],image_dir:3,image_format:12,image_s:2,image_shap:12,imagesaverhook:3,imagespec:2,import_modul:16,index:[2,7],initi:[6,8,13],input_fn:3,ioio:16,job:6,jpeg:10,label:[1,2,11],labelspec:2,learning_r:5,list:0,list_modul:16,log:16,log_parse_arg:16,loss:[1,5],loss_fn:1,main:6,make:0,map_dict:16,map_fn:16,minim:1,mode:11,model:[1,4,8],model_cl:1,model_fn:11,model_param:1,model_pkg:4,modeliniti:4,modul:[7,16],multilabelspec:2,multipl:1,name:[0,16],name_scop:16,ndarrai:10,nested_dict:16,none:[0,1,2,3,5,12,16],normal:12,num_parallel_cal:0,numpi:10,oper:1,ops:8,optim:[1,8,15],original_dict:16,original_fn:16,over:1,overwrit:16,packag:7,page:7,param:[6,11,16],paramet:[0,6,10],parent_kei:2,pars:2,parse_arg:[4,11,16],parser:0,path:2,pkg:16,process_fn:0,proto:2,read:0,record:2,refer:7,resolv:6,run:[3,6],run_context:3,run_valu:3,runner:[4,8],save_path:16,search:7,session:[3,8],setter:16,shape:2,sourc:[0,1,2,3,4,5,6,10,11,12,14,16],spec:2,specifi:2,sphinx:3,split:0,store:0,str:[0,10,16],string:[2,10],strip:16,strip_dict_arg:16,strip_illegal_summary_nam:16,summari:[3,16],summary_dir:3,summary_op:3,tensor:2,test:0,test_siz:0,tflib:[0,1,2,3,4,5,6],tfrecord:0,tfrecord_filenam:0,tmp:6,tower:1,tower_grad:5,train:[0,1,3,5,8],train_it:5,train_op:5,traininiti:4,uniqu:2,unknown:4,url:16,used:0,util:8,value_dict:2,var_list:5,var_scop:5,where:0,write:0},titles:["Dataset","Dispatcher","FeatureSpec","Hooks","Initializers","Optimizer","Runner","Welcome to tflibs\u2019s documentation!","tflibs package","tflibs.datasets","tflibs.image","tflibs.model","tflibs.ops","tflibs.runner","tflibs.session","tflibs.training","tflibs.utils package"],titleterms:{dataset:[0,9],dispatch:1,document:7,featurespec:2,hook:3,imag:[10,12],indic:7,initi:4,model:11,ops:12,optim:5,packag:[8,16],refer:8,runner:[6,13],session:14,tabl:7,tflib:[7,8,9,10,11,12,13,14,15,16],tftool:[],train:15,util:[10,16],welcom:7}})