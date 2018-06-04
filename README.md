# FINAL-ANALYSIS

the folder is composed by 2 files , 1 jupyter notebook (1) -demo Influencers Kingcom- and 1 python script (2) -modules_for_demo-.  

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
