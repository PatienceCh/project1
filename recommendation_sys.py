# importing libraries
import streamlit as st
import pickle
import sklearn
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from PIL import Image

# Some app designing
st.set_page_config(
    page_title ='High School Subject Recommendation System',
    layout = 'wide',
    initial_sidebar_state = 'expanded')
st.title('Mid-level High School Subject Recommendation System')
image = Image.open('Academic Success.jpg')
st.image(image, width=500)
st.sidebar.header("SUBJECT MARK ENTRY FORM")


# load data

df = pd.read_csv('Clean_Transformed_Pupil_Record.csv')


### MATRIX OF SUBJECTS AND STUDENTS
# The data represents individual subjects taken by students and for the recommendation system with collaborative filtering algorithm, 
# the data must have each record containing information about which subject each student has taken.

# creating a matrix of students and subjects where each row represents the marks attained in each subject by each student.
stud_subj_df1 = df.pivot_table(
    index = 'stuid',
    columns = 'subject',
    values = 'mark')

# averaging termly attainments into yearly average attainments
round_fig=2 # rounding value

# yearly average attainments for form one
stud_subj_df1['ART_F1'] = np.round((stud_subj_df1[['ARTF1T1', 'ARTF1T2', 'ARTF1T3']].mean(axis = 1)),round_fig)
stud_subj_df1['COMP SC_F1'] = np.round((stud_subj_df1[['COMPF1T1', 'COMPF1T2', 'COMPF1T3']].mean(axis = 1)),round_fig)
stud_subj_df1['ELA_F1'] = np.round((stud_subj_df1[['ELAF1T1', 'ELAF1T2', 'ELAF1T3']].mean(axis = 1)),round_fig)
stud_subj_df1['FRE_F1'] = np.round((stud_subj_df1[['FREF1T1', 'FREF1T2', 'FREF1T3']].mean(axis = 1)),round_fig)
stud_subj_df1['GEO_F1'] = np.round((stud_subj_df1[['GEOF1T1', 'GEOF1T2', 'GEOF1T3']].mean(axis = 1)),round_fig)
stud_subj_df1['HIS_F1'] = np.round((stud_subj_df1[['HISF1T1', 'HISF1T2', 'HISF1T3']].mean(axis = 1)),round_fig)
stud_subj_df1['MTH_F1'] = np.round((stud_subj_df1[['MTHF1T1', 'MTHF1T2', 'MTHF1T3']].mean(axis = 1)),round_fig)
stud_subj_df1['RE_F1'] = np.round((stud_subj_df1[['REF1T1', 'REF1T2', 'REF1T3']].mean(axis = 1)),round_fig)
stud_subj_df1['SCI_F1'] = np.round((stud_subj_df1[['SCIF1T1', 'SCIF1T2', 'SCIF1T3']].mean(axis = 1)),round_fig)
stud_subj_df1['SH_F1'] = np.round((stud_subj_df1[['SHF1T1', 'SHF1T2', 'SHF1T3']].mean(axis = 1)),round_fig)

# yearly average attainments for form two
stud_subj_df1['ART_F2'] = np.round((stud_subj_df1[['ARTF2T1', 'ARTF2T2', 'ARTF2T3']].mean(axis = 1)),round_fig)
stud_subj_df1['COMP SC_F2'] = np.round((stud_subj_df1[['COMPF2T1', 'COMPF2T2', 'COMPF2T3']].mean(axis = 1)),round_fig)
stud_subj_df1['ELA_F2'] = np.round((stud_subj_df1[['ELAF2T1', 'ELAF2T2', 'ELAF2T3']].mean(axis = 1)),round_fig)
stud_subj_df1['FRE_F2'] = np.round((stud_subj_df1[['FREF2T1', 'FREF2T2', 'FREF2T3']].mean(axis = 1)),round_fig)
stud_subj_df1['GEO_F2'] = np.round((stud_subj_df1[['GEOF2T1', 'GEOF2T2', 'GEOF2T3']].mean(axis = 1)),round_fig)
stud_subj_df1['HIS_F2'] = np.round((stud_subj_df1[['HISF2T1', 'HISF2T2', 'HISF2T3']].mean(axis = 1)),round_fig)
stud_subj_df1['MTH_F2'] = np.round((stud_subj_df1[['MTHF2T1', 'MTHF2T2', 'MTHF2T3']].mean(axis = 1)),round_fig)
stud_subj_df1['RE_F2'] = np.round((stud_subj_df1[['REF2T1', 'REF2T2', 'REF2T3']].mean(axis = 1)),round_fig)
stud_subj_df1['SCI_F2'] = np.round((stud_subj_df1[['SCIF2T1', 'SCIF2T2', 'SCIF2T3']].mean(axis = 1)),round_fig)
stud_subj_df1['SH_F2'] = np.round((stud_subj_df1[['SHF2T1', 'SHF2T2', 'SHF2T3']].mean(axis = 1)),round_fig)

# yearly average attainments for form three and form four
stud_subj_df1['ACC_F3&4'] = np.round((stud_subj_df1[['ACCF3T1', 'ACCF3T2', 'ACCF3T3', 'ACCF4T1','ACCF4T2']].mean(axis = 1)),round_fig)
stud_subj_df1['ART_F3&4'] = np.round((stud_subj_df1[['ARTF3T1', 'ARTF3T2', 'ARTF3T3', 'ARTF4T1','ARTF4T2']].mean(axis = 1)),round_fig)
stud_subj_df1['BIOL_F3&4'] = np.round((stud_subj_df1[['BIOLF3T1', 'BIOLF3T2', 'BIOLF3T3', 'BIOLF4T1','BIOLF4T2']].mean(axis = 1)),round_fig)
stud_subj_df1['BST_F3&4'] = np.round((stud_subj_df1[['BSTF3T1', 'BSTF3T2', 'BSTF3T3', 'BSTF4T1','BSTF4T2']].mean(axis = 1)),round_fig)
stud_subj_df1['CHEM_F3&4'] = np.round((stud_subj_df1[['CHEMF3T1', 'CHEMF3T2', 'CHEMF3T3', 'CHEMF4T1','CHEMF4T2']].mean(axis = 1)),round_fig)
stud_subj_df1['COMP SC_F3&4'] = np.round((stud_subj_df1[['COMP SCF3T1', 'COMP SCF3T2', 'COMP SCF3T3', 'COMP SCF4T1','COMP SCF4T2']].mean(axis = 1)),round_fig)
stud_subj_df1['D&T_F3&4'] = np.round((stud_subj_df1[['D&TF3T1', 'D&TF3T2', 'D&TF3T3', 'D&TF4T1','D&TF4T2']].mean(axis = 1)),round_fig)
stud_subj_df1['ELA_F3&4'] = np.round((stud_subj_df1[['ELAF3T1', 'ELAF3T2', 'ELAF3T3', 'ELAF4T1','ELAF4T2']].mean(axis = 1)),round_fig)
stud_subj_df1['ELIT_F3&4'] = np.round((stud_subj_df1[['ELITF4T1','ELITF4T2']].mean(axis = 1)),round_fig)
stud_subj_df1['FRE_F3&4'] = np.round((stud_subj_df1[['FREF3T1', 'FREF3T2', 'FREF3T3', 'FREF4T1','FREF4T2']].mean(axis = 1)),round_fig)
stud_subj_df1['GEO_F3&4'] = np.round((stud_subj_df1[['GEOF3T1', 'GEOF3T2', 'GEOF3T3', 'GEOF4T1','GEOF4T2']].mean(axis = 1)),round_fig)
stud_subj_df1['HIS_F3&4'] = np.round((stud_subj_df1[['HISF3T1', 'HISF3T2', 'HISF3T3', 'HISF4T1','HISF4T2']].mean(axis = 1)),round_fig)
stud_subj_df1['ICT_F3&4'] = np.round((stud_subj_df1[['ICTF3T1', 'ICTF3T2', 'ICTF3T3', 'ICTF4T1','ICTF4T2']].mean(axis = 1)),round_fig)
stud_subj_df1['MTH_F3&4'] = np.round((stud_subj_df1[['MTHF3T1', 'MTHF3T2', 'MTHF3T3', 'MTHF4T1','MTHF4T2']].mean(axis = 1)),round_fig)
stud_subj_df1['PED_F3&4'] = np.round((stud_subj_df1[['PEDF3T1', 'PEDF3T2', 'PEDF3T3', 'PEDF4T1','PEDF4T2']].mean(axis = 1)),round_fig)
stud_subj_df1['PHY_F3&4'] = np.round((stud_subj_df1[['PHYF3T1', 'PHYF3T2', 'PHYF3T3', 'PHYF4T1','PHYF4T2']].mean(axis = 1)),round_fig)
stud_subj_df1['RE_F3&4'] = np.round((stud_subj_df1[['REF3T1', 'REF3T2', 'REF3T3', 'REF4T1','REF4T2']].mean(axis = 1)),round_fig)

# creating a dataframe with yearly average attainments
stud_subj_df2 = stud_subj_df1[['ART_F1','COMP SC_F1','ELA_F1','FRE_F1','GEO_F1','HIS_F1','MTH_F1','RE_F1','SCI_F1',
                               'SH_F1','ART_F2','COMP SC_F2','ELA_F2','FRE_F2','GEO_F2','HIS_F2','MTH_F2','RE_F2',
                               'SCI_F2','SH_F2','ACC_F3&4','ART_F3&4','BIOL_F3&4','BST_F3&4','CHEM_F3&4','COMP SC_F3&4',
                               'D&T_F3&4','ELA_F3&4','ELIT_F3&4','FRE_F3&4','GEO_F3&4','HIS_F3&4','ICT_F3&4','MTH_F3&4',
                               'PED_F3&4','PHY_F3&4','RE_F3&4']]


# creating an input form for User
col1, col2 = st.columns(2)

with col1:
    st.sidebar.header("Form One Subjects")
    with st.sidebar.form('form1', clear_on_submit=True):
        st.subheader('Term-1 Subjects')
        ART_F1t1= st.number_input("F1t1_ART",)
        COMPSC_F1t1 = st.number_input("F1t1_COMPUTER SCIENCE",)
        ELA_F1t1 = st.number_input("F1t1_ENGLISH LANGUAGE",)
        FRE_F1t1 = st.number_input("F1t1_FRENCH",)
        GEO_F1t1 = st.number_input("F1t1_GEOGRAPHY",)
        HIS_F1t1 = st.number_input("F1t1_HISTORY",)
        MTH_F1t1 = st.number_input("F1t1_MATHEMATICS",)
        RE_F1t1 = st.number_input("F1t1_RELIGIOUS EDUCATION",)
        SCI_F1t1 = st.number_input("F1t1_SCIENCE",)
        SH_F1t1 = st.number_input("F1t1_SHONA",)
        
        st.subheader('Term-2 Subjects')
        ART_F1t2= st.number_input("F1t2_ART",)
        COMPSC_F1t2 = st.number_input("F1t2_COMPUTER SCIENCE",)
        ELA_F1t2 = st.number_input("F1t2_ENGLISH LANGUAGE",)
        FRE_F1t2 = st.number_input("F1t2_FRENCH",)
        GEO_F1t2 = st.number_input("F1t2_GEOGRAPHY",)
        HIS_F1t2 = st.number_input("F1t2_HISTORY",)
        MTH_F1t2 = st.number_input("F1t2_MATHEMATICS",)
        RE_F1t2 = st.number_input("F1t2_RELIGIOUS EDUCATION",)
        SCI_F1t2 = st.number_input("F1t2_SCIENCE",)
        SH_F1t2 = st.number_input("F1t2_SHONA",)
        
        
        st.subheader('Term-3 Subjects')
        ART_F1t3= st.number_input("F1t3_ART",)
        COMPSC_F1t3 = st.number_input("F1t3_COMPUTER SCIENCE",)
        ELA_F1t3 = st.number_input("F1t3_ENGLISH LANGUAGE",)
        FRE_F1t3 = st.number_input("F1t3_FRENCH",)
        GEO_F1t3 = st.number_input("F1t3_GEOGRAPHY",)
        HIS_F1t3 = st.number_input("F1t3_HISTORY",)
        MTH_F1t3 = st.number_input("F1t3_MATHEMATICS",)
        RE_F1t3 = st.number_input("F1t3_RELIGIOUS EDUCATION",)
        SCI_F1t3 = st.number_input("F1t3_SCIENCE",)
        SH_F1t3 = st.number_input("F1t3_SHONA",)
        submitted = st.form_submit_button('Submit')
    

with col2:
    st.sidebar.header("Form Two Subjects")
    with st.sidebar.form('form2', clear_on_submit=True):
        st.subheader('Term-1 Subjects')
        ART_F2t1 = st.number_input("F2t1_ART",)
        COMPSC_F2t1 = st.number_input("F2t1_COMPUTER SCIENCE",)
        ELA_F2t1 = st.number_input("F2t1_ENGLISH LANGUAGE",)
        FRE_F2t1 = st.number_input("F2t1_FRENCH",)
        GEO_F2t1 = st.number_input("F2t1_GEOGRAPHY",)
        HIS_F2t1 = st.number_input("F2t1_HISTORY",)
        MTH_F2t1 = st.number_input("F2t1_MATHEMATICS",)
        RE_F2t1 = st.number_input("F2t1_RELIGIOUS EDUCATION",)
        SCI_F2t1 = st.number_input("F2t1_SCIENCE",)
        SH_F2t1 = st.number_input("F2t1_SHONA",)
        
        st.subheader('Term-2 Subjects')
        ART_F2t2 = st.number_input("F2t2_ART",)
        COMPSC_F2t2 = st.number_input("F2t2_COMPUTER SCIENCE",)
        ELA_F2t2 = st.number_input("F2t2_ENGLISH LANGUAGE",)
        FRE_F2t2 = st.number_input("F2t2_FRENCH",)
        GEO_F2t2 = st.number_input("F2t2_GEOGRAPHY",)
        HIS_F2t2 = st.number_input("F2t2_HISTORY",)
        MTH_F2t2 = st.number_input("F2t2_MATHEMATICS",)
        RE_F2t2 = st.number_input("F2t2_RELIGIOUS EDUCATION",)
        SCI_F2t2 = st.number_input("F2t2_SCIENCE",)
        SH_F2t2 = st.number_input("F2t2_SHONA",)
        
        st.subheader('Term-3 Subjects')       
        ART_F2t3 = st.number_input("F2t3_ART",)
        COMPSC_F2t3 = st.number_input("F2t3_COMPUTER SCIENCE",)
        ELA_F2t3 = st.number_input("F2t3_ENGLISH LANGUAGE",)
        FRE_F2t3 = st.number_input("F2t3_FRENCH",)
        GEO_F2t3 = st.number_input("F2t3_GEOGRAPHY",)
        HIS_F2t3 = st.number_input("F2t3_HISTORY",)
        MTH_F2t3 = st.number_input("F2t3_MATHEMATICS",)
        RE_F2t3 = st.number_input("F2t3_RELIGIOUS EDUCATION",)
        SCI_F2t3 = st.number_input("F2t3_SCIENCE",)
        SH_F2t3 = st.number_input("F2t3_SHONA",)
        submitted = st.form_submit_button('Submit')
        
# Calculating average subject marks obtained for form 1 and form 2
periods=3 # number of terms per year
round_fig=2 # rounding value

ART_F1 =np.round(((ART_F1t1+ART_F1t2+ART_F1t3)/periods),round_fig)
COMPSC_F1 = np.round(((COMPSC_F1t1+COMPSC_F1t2+COMPSC_F1t3)/periods),round_fig)
ELA_F1 = np.round(((ELA_F1t1+ELA_F1t3+ELA_F1t3)/periods),round_fig)
FRE_F1 = np.round(((FRE_F1t1+FRE_F1t2+FRE_F1t3)/periods),round_fig)
GEO_F1 = np.round(((GEO_F1t1+GEO_F1t2+GEO_F1t3)/periods),round_fig)
HIS_F1 = np.round(((HIS_F1t1+HIS_F1t2+HIS_F1t3)/periods),round_fig)
MTH_F1 = np.round(((MTH_F1t1+MTH_F1t2+MTH_F1t3)/periods),round_fig)
RE_F1 = np.round(((RE_F1t1+RE_F1t2+RE_F1t3)/periods),round_fig)
SCI_F1 = np.round(((SCI_F1t1+SCI_F1t2+SCI_F1t3)/periods),round_fig)
SH_F1 = np.round(((SH_F1t1+SH_F1t2+SH_F1t3)/periods),round_fig)
        
ART_F2 = np.round(((ART_F2t1+ART_F2t2+ART_F2t3)/periods),round_fig)
COMPSC_F2 = np.round(((COMPSC_F2t1+COMPSC_F2t2+COMPSC_F2t3)/periods),round_fig)
ELA_F2 = np.round(((ELA_F2t1+ELA_F2t2+ELA_F2t3)/periods),round_fig)
FRE_F2 = np.round(((FRE_F2t1+FRE_F2t2+FRE_F2t3)/periods),round_fig)
GEO_F2 = np.round(((GEO_F2t1+GEO_F2t2+GEO_F2t3)/periods),round_fig)
HIS_F2 = np.round(((HIS_F2t1+HIS_F2t2+HIS_F2t3)/periods),round_fig)
MTH_F2 = np.round(((MTH_F2t1+MTH_F2t2+MTH_F2t3)/periods),round_fig)
RE_F2 = np.round(((RE_F2t1+RE_F2t2+RE_F2t3)/periods),round_fig)
SCI_F2 = np.round(((SCI_F2t1+SCI_F2t2+SCI_F2t3)/periods),round_fig)
SH_F2 = np.round(((SH_F2t1+SH_F2t2+SH_F2t3)/periods),round_fig)        
        


# creating dataframe with form one and two yearly average attainments for use in computing pairwise distances 
stud_subj_df3 = stud_subj_df2[['ART_F1','COMP SC_F1','ELA_F1','FRE_F1','GEO_F1','HIS_F1','MTH_F1',
                                                     'RE_F1','SCI_F1','SH_F1','ART_F2','COMP SC_F2','ELA_F2','FRE_F2','GEO_F2',
                                                     'HIS_F2','MTH_F2','RE_F2','SCI_F2','SH_F2']]

# Adding new form 3 student's data to the matrix of students and subjects
stud_subj_df4 = stud_subj_df3.copy()
stud_subj_df4.loc[len(stud_subj_df4)]=pd.Series({'ART_F1':ART_F1,'COMP SC_F1':COMPSC_F1,'ELA_F1':ELA_F1,
                                                                     'FRE_F1':FRE_F1,'GEO_F1':GEO_F1,'HIS_F1':HIS_F1,'MTH_F1':MTH_F1,
                                                                     'RE_F1':RE_F1,'SCI_F1':SCI_F1,'SH_F1':SH_F1,'ART_F2':ART_F2,
                                                                     'COMP SC_F2':COMPSC_F2,'ELA_F2':ELA_F2,'FRE_F2':FRE_F2,
                                                                     'GEO_F2':GEO_F2,'HIS_F2':HIS_F2,'MTH_F2':MTH_F2,'RE_F2':RE_F2,
                                                                     'SCI_F2':SCI_F2,'SH_F2':SH_F2})

stud_subj_df4.rename(index = {579:'User'}, inplace = True)


# Displaying the new student data entries
st.subheader('Student Data')
st.dataframe(stud_subj_df4.tail(1))
    
#### Using euclidean_distances function to calculate the differences in pairs between the students, and creating a dataframe with this output array and store it in a variable called student_student_dist_matrix, which means student_student distance matrix

### COLLABORATIVE FILTERING
#  finding similarities with new form 3 student
stud_stud_dist_df = pd.DataFrame(euclidean_distances(stud_subj_df4))

# Renaming the index and columns of the dataframe as each row and column in the index represents indivdual students making hard to understand.
stud_stud_dist_df.columns = stud_subj_df4.index
stud_stud_dist_df['stuid'] = stud_subj_df4.index
stud_stud_dist_df = stud_stud_dist_df.set_index('stuid')

#### Using the student_student_dist_matrix we can easily tell which students are similar to others, and which students have taken similar subjects from others.  

# Identifying similar students to the User (new_form3_student) based on the euclidian distance 
sim_stud_df = pd.DataFrame(stud_stud_dist_df.loc[stud_subj_df4.index[-1]].sort_values(ascending = False))
sim_stud_df = sim_stud_df.reset_index()


# Identifying the 5 most similar students to the User based on the euclidian distance
# Dropped record zero because it is the User's euclidian distance. 
most_sim_stud_df = sim_stud_df.drop(sim_stud_df.tail(5).index[4]).tail(5)
most_sim_stud_df = most_sim_stud_df.reset_index()
most_sim_stud_df = most_sim_stud_df.drop(['index'], axis = 1)

#### Using 1-d_norm function to convert the calculated distances in pairs between the students, and creating a dataframe  variable called student_student_sim_matrix, which means student_student similarity matrix. To make subject recommendations, the similarity scores are used.
most_sim_stud_df['Sim_score'] = 1-(most_sim_stud_df['User']/sim_stud_df['User'].max())
most_sim_stud_df = most_sim_stud_df.sort_values('Sim_score', ascending = False)
most_sim_stud_df.rename(columns={'User': 'Ecl_distance'}, inplace=True)

# Displaying the most_similar_students

st.subheader('Similarity score with 5 Most Similar Students')
st.write("========================================================================")
if stud_subj_df4.tail(1).loc['User'].sum()==0:
    st.write("Similarity score with Top most similar student (max_sim): ", '{:,.2%}'.format(0))
    st.write("Similarity score with Bottom most similar student (min_sim): ",'{:,.2%}'.format(0))
    st.write("Average Similarity score with 5 Most Similar Students (avg_sim): ", '{:,.2%}'.format(0))
elif stud_subj_df4.tail(1).loc['User'].sum()>0:
    st.write("Similarity score with Top most similar student (max_sim): ", '{:,.2%}'.format(most_sim_stud_df.iloc[:,2].max()))
    st.write("Similarity score with Bottom most similar student (min_sim): ",'{:,.2%}'.format(most_sim_stud_df.iloc[:,2].min()))
    st.write("Average Similarity score with 5 Most Similar Students (avg_sim): ", '{:,.2%}'.format(most_sim_stud_df.iloc[:,2].sum()/len(most_sim_stud_df.iloc[:,2])))
st.write("========================================================================")

# retrieving the subjects that the User has taken and passed on average using the 'nonzero' function.
subj_passedby_user = set(stud_subj_df4.applymap(
    lambda x: 1 if x >49.99 else 0).loc[stud_subj_df4.index[-1]].iloc[stud_subj_df4.applymap(
    lambda x: 1 if x >49.99 else 0).loc[stud_subj_df4.index[-1]].to_numpy().nonzero()].index)

# retrieving the subjects that the "most_similar_student(s)" has taken and passed on average using the 'nonzero' function
y = [set(stud_subj_df2.applymap(lambda x: 1 if x >49.99 else 0).loc[i].iloc[ stud_subj_df2.applymap(
    lambda x: 1 if x >49.99 else 0).loc[i].to_numpy().nonzero()].index) for i in most_sim_stud_df['stuid']]
subj_passedby_mostsim_stud= y[0].union(y[1],y[2],y[3],y[4]) 

# We have two sets of subjects taken and passed by the User and "most_similar_student(s)". 
# Set operation is used to find subjects taken by "most_similar_student(s)" but haven't been taken or passed by the User

# MAKING RECOMMENDATIONS
# A set of form 1 and 2 subjects
form1_2_subj = ['ART_F1','COMP SC_F1','ELA_F1','FRE_F1','GEO_F1','HIS_F1','MTH_F1','RE_F1','SCI_F1','SH_F1','ART_F2',
                        'COMP SC_F2','ELA_F2','FRE_F2','GEO_F2','HIS_F2','MTH_F2','RE_F2','SCI_F2','SH_F2']
form1_2_subj = set(form1_2_subj)
    
# science subject at form 1 and 2
Lower_form_sci = ['SCI_F1','SCI_F2']
Lower_form_sci = set(Lower_form_sci)
    
# A set of core subject at form 3 and 4
core_subj = ['MTH_F3&4','ELA_F3&4','RE_F3&4']
core_subj = set(core_subj)

# humanities and commercial subjects at form 3 and 4
hum_comm_subj = ['ACC_F3&4','BST_F3&4','ART_F3&4','ELIT_F3&4','FRE_F3&4','GEO_F3&4','HIS_F3&4','PED_F3&4']
hum_comm_subj = set(hum_comm_subj)

# science subjects at form 3 and 4
midlevel_sci_subj = ['BIOL_F3&4','CHEM_F3&4','COMP SC_F3&4','D&T_F3&4','ICT_F3&4','PHY_F3&4']
midlevel_sci_subj = set(midlevel_sci_subj)

# subjects to recommend to new form 3 student
subj_to_recom = (subj_passedby_mostsim_stud-subj_passedby_user-form1_2_subj-core_subj)


# subjects_to_recommend_to_new_form3_student
core = list(core_subj)
for i in range(len(core_subj)):
    if core[i]=='MTH_F3&4':
        core[i]='MATHEMATICS'
    elif core[i]=='ELA_F3&4':
        core[i]='ENGLISH LANGUAGE'
    elif core[i]=='RE_F3&4':
        core[i]='RELIGIOUS STUDIES'

hum_comm = list(hum_comm_subj)
for i in range(len(hum_comm_subj)):
    if hum_comm[i]=='ACC_F3&4':
        hum_comm[i]='ACCOUNTING'
    elif hum_comm[i]=='ART_F3&4':
        hum_comm[i]='ART'
    elif hum_comm[i]=='BST_F3&4':
        hum_comm[i]='BUSINESS STUDIES'
    elif hum_comm[i]=='ELIT_F3&4':
        hum_comm[i]='ENGLISH LITERATURE'
    elif hum_comm[i]=='FRE_F3&4':
        hum_comm[i]='FRENCH'
    elif hum_comm[i]=='GEO_F3&4':
        hum_comm[i]='GEOGRAPHY'
    elif hum_comm[i]=='HIS_F3&4':
        hum_comm[i]='HISTORY'
    elif hum_comm[i]=='PED_F3&4':
        hum_comm[i]='PHYSICAL EDUCATION'
        
sci = list(midlevel_sci_subj)
for i in range(len(midlevel_sci_subj)):
    if sci[i]=='BIOL_F3&4':
        sci[i]='BIOLOGY'
    elif sci[i]=='CHEM_F3&4':
        sci[i]='CHEMISTRY'
    elif sci[i]=='COMP SC_F3&4':
        sci[i]='COMPUTER SCIENCE'
    elif sci[i]=='D&T_F3&4':
        sci[i]='DESING AND TECHNOLOGY'
    elif sci[i]=='ICT_F3&4':
        sci[i]='INFORMATION COMMUNICATION TECHNOLOGY'
    elif sci[i]=='PHY_F3&4':
        sci[i]='PHYSICS'
        
electives = list(subj_to_recom)
for i in range(len(subj_to_recom)):
    if electives[i]=='ACC_F3&4':
        electives[i]='ACCOUNTING'
    elif electives[i]=='ART_F3&4':
        electives[i]='ART'
    elif electives[i]=='BIOL_F3&4':
        electives[i]='BIOLOGY'
    elif electives[i]=='BST_F3&4':
        electives[i]='BUSINESS STUDIES'
    elif electives[i]=='CHEM_F3&4':
        electives[i]='CHEMISTRY'
    elif electives[i]=='COMP SC_F3&4':
        electives[i]='COMPUTER SCIENCE'
    elif electives[i]=='D&T_F3&4':
        electives[i]='DESING AND TECHNOLOGY'
    elif electives[i]=='ELIT_F3&4':
        electives[i]='ENGLISH LITERATURE'
    elif electives[i]=='FRE_F3&4':
        electives[i]='FRENCH'
    elif electives[i]=='GEO_F3&4':
        electives[i]='GEOGRAPHY'
    elif electives[i]=='HIS_F3&4':
        electives[i]='HISTORY'
    elif electives[i]=='ICT_F3&4':
        electives[i]='INFORMATION COMMUNICATION TECHNOLOGY'
    elif electives[i]=='PED_F3&4':
        electives[i]='PHYSICAL EDUCATION'
    elif electives[i]=='PHY_F3&4':
        electives[i]='PHYSICS'


if st.button('Show Recommendations', type ='primary'):
    st.subheader("Recommended Subjects")
    st.write("========================================================================")
    if stud_subj_df4.tail(1).loc['User'].sum()==0:
        st.write("Please enter your marks on the subject marks entry form!!!!")
        
    elif stud_subj_df4.tail(1).loc['User'].sum()>0:
        st.subheader("Core Subjects")
        st.write(core)
        st.write(".........................................................................................................")
        st.subheader("Elective Subjects")
        
        if len(electives)<7 or len(Lower_form_sci.intersection(subj_passedby_user))==0:
            electives = list((set(electives).union(set(hum_comm)))-set(sci))
            st.write(electives)
            
        elif len(electives)<7 or len(Lower_form_sci.intersection(subj_passedby_user))>0:
            electives = list((set(electives).union(set(hum_comm))))
            st.write(electives)
            
        else:    
            st.write(electives)
        st.write("...........................................................................................................")

# function to reset the student_subject_df3
stud_subj_df4=stud_subj_df4.drop(stud_subj_df4.index[-1], axis = 0)