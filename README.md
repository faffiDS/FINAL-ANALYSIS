# FINAL-ANALYSIS

the folder is composed by 2 files (for just french space and for internazionalized space (i18n) , 1 jupyter notebook (1) -demo Influencers Kingcom- and 1 python script (2) -modules_for_demo-.  

(1) is the main script for the final analysis , you can run it for any topic and have the final map

(2) is the script containing the functions to be called for running the main .


NB in (2) :

#matrix is the vector space created during the previous script  
#"matrix_user_vector_facounnier_200" is the name of the csv file loaded into pandas dataframe  
matrix = pd.read_csv('matrix_user_vector_facounnier_200_more_people.csv',sep='\t',index_col=0)  
#it is the word2vec dictionary , the .bin file can be downloaded (as for the following file) or trained with gensim  
file_where_dictionary_is_stored = "/home/ubuntu/mynotebooks/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin"    

You can replace "matrix_user_vector_facounnier_200_more_people.csv" with a new matrix trained by means of "BUILDING VECTOR SPACE" script  
and  

You can replace "home/ubuntu/mynotebooks/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin" with another model that can be downloaded or trained by means of "BUILDING+VECTOR+SPACE"  


line 242 in modules_for_demo.py:  
 q = select(a , 0.9, 0.3 , 0.3 ,0.3 ).sort_values('final_measure' , ascending=False).head(100).round({'final_measure': 2, 'likesperday': 0})  
 
 here You can increase or decrease quantiles for the influencers research.  
 q1 (0,9)  = final measure (or topic closeness measure)   
 q2 (0,3)  = likes  
 q3 (0,3)  = followers   
 q4 (0,3)  = likes per day    
 increase > smaller bunch of influencers   
 decrease > vice versa  
 
 THE SAME IDEA FOR I18N+demo+Influencers+-+Kingcom.ipynb AND  	modules_for_demo_i18n.py  
 
 THE LAST SCRIPT IS : 
  FINAL-ANALYSIS/tool+that+takes+into+account+texts+and+photos+on+the+150+users++from+raw+data+to+final+product.ipynb   
  
  this script is a kind of prototype that takes care of texts and photos too. It contains further research about importance of texts or photos .  
  It contains everything from the space building to the final map without modules  
  NB 
  cell 2 
  en_dictionary = gensim.models.KeyedVectors.load_word2vec_format('/home/ubuntu/mynotebooks/GoogleNews-vectors-negative300.bin',binary=True) , you can download it here : https://github.com/mmihaltz/word2vec-GoogleNews-vectors  
  
  cell 4  
  photo_list = glob.glob("/home/ubuntu/mynotebooks/dati_foto_dp/dati_foto_dp/*/") , the folder that contains Amazon Rekognition tags for photos , obviously replace this path with yours  
  the ground truth (cell 29) is made by me to give veracity to the tool
 
