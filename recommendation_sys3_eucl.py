# importing libraries
import streamlit as st
import pickle
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
data = pickle.load(open('Student_Records.sav','rb'))

# creating a matrix of students and subjects 
student_subject_matrix = data.pivot_table(
    index = 'Stuid',
    columns = 'Subject_Taken',
    values = 'AvgMarkObtained')

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
        
### MATRIX OF SUBJECTS AND STUDENTS
# The data represents individual subjects taken by students and for the recommendation system with collaborative filtering algorithm, 
# the data must have each record containing information about which subject each student has taken.
#### Making a matrix where each row represents the marks attained in each subject by each student. 

student_subject_matrix1 = student_subject_matrix[['ART_F1','COMP SC_F1','ELA_F1','FRE_F1','GEO_F1','HIS_F1','MTH_F1',
                                                     'RE_F1','SCI_F1','SH_F1','ART_F2','COMP SC_F2','ELA_F2','FRE_F2','GEO_F2',
                                                     'HIS_F2','MTH_F2','RE_F2','SCI_F2','SH_F2']]

# Adding new form 3 student's data to the matrix of students and subjects
student_subject_matrix2 = student_subject_matrix1.copy()
student_subject_matrix2.loc[len(student_subject_matrix2)]=pd.Series({'ART_F1':ART_F1,'COMP SC_F1':COMPSC_F1,'ELA_F1':ELA_F1,
                                                                     'FRE_F1':FRE_F1,'GEO_F1':GEO_F1,'HIS_F1':HIS_F1,'MTH_F1':MTH_F1,
                                                                     'RE_F1':RE_F1,'SCI_F1':SCI_F1,'SH_F1':SH_F1,'ART_F2':ART_F2,
                                                                     'COMP SC_F2':COMPSC_F2,'ELA_F2':ELA_F2,'FRE_F2':FRE_F2,
                                                                     'GEO_F2':GEO_F2,'HIS_F2':HIS_F2,'MTH_F2':MTH_F2,'RE_F2':RE_F2,
                                                                     'SCI_F2':SCI_F2,'SH_F2':SH_F2})


# Displaying the new student data entries
st.subheader('Student Data')
st.dataframe(student_subject_matrix2.tail(1))
    
# Using  euclidean_distances function to calculate the similarities in pairs between the students, 
# and creating a dataframe with this output array and store it in a variable called student_student_sim_matrix, 
# which means student_student similarity matrix

### COLLABORATIVE FILTERING
#  finding similarities with new form 3 student

student_student_sim_matrix = pd.DataFrame(euclidean_distances(student_subject_matrix2))
# Renaming the index and columns of the dataframe as each row and column in the index represents indivdual students making hard to understand.
student_student_sim_matrix.columns = student_subject_matrix2.index
student_student_sim_matrix['Stuid'] = student_subject_matrix2.index
student_student_sim_matrix = student_student_sim_matrix.set_index('Stuid')

# Using the student_student_sim_matrix we can easily tell which students are similar to others, 
# and which students have taken similar subjects from others. 
# To make subject recommendations, the  euclidean_distances measures are used.

# identifying the 10 most similar students to the "new_form3_student" based on the  euclidean_distances
most_similar_students = pd.DataFrame(student_student_sim_matrix.loc[student_subject_matrix2.index[-1]].sort_values(ascending = False))
most_similar_students = most_similar_students.reset_index()
# Dropped record 10 because it is the "new_form3_student's" similarity record. 
most_similar_students = most_similar_students.drop(most_similar_students.tail(10).index[9]).tail(10)
most_similar_students = most_similar_students.reset_index()
most_similar_students = most_similar_students.drop(['index'], axis = 1)

# Displaying the most_similar_students

st.subheader('Similarity distance with 10 Most Similar Students')
st.write("========================================================================")
if student_subject_matrix2.tail(1).loc[579].sum()==0:
    st.write("Top Most Similar Student (min_dist): ", 0.00)
    st.write("Bottom Most Similar Student (max_dist): ", 0.00)
    st.write("Average Similarity distance with 10 Most Similar Students (avg_dist): ", 0.00)
elif student_subject_matrix2.tail(1).loc[579].sum()>0:
    st.write("Top Most Similar Student (min_dist): ", np.round((most_similar_students.iloc[:,1].min()),round_fig))
    st.write("Bottom Most Similar Student (max_dist): ", np.round((most_similar_students.iloc[:,1].max()),round_fig))
    st.write("Average Similarity distance with 10 Most Similar Students (avg_dist): ",
             np.round((most_similar_students.iloc[:,1].sum()/len(most_similar_students.iloc[:,1])),round_fig))
st.write("========================================================================")

# retrieving the subjects that the "new_form3_student" has taken and passed on average using the 'nonzero' function.
subj_passedby_newf3stu = set(student_subject_matrix2.applymap(
    lambda x: 1 if x >49.99 else 0).loc[student_subject_matrix2.index[-1]].iloc[student_subject_matrix2.applymap(
    lambda x: 1 if x >49.99 else 0).loc[student_subject_matrix2.index[-1]].to_numpy().nonzero()].index)

# retrieving the subjects that the "most_similar_student(s)" has taken and passed on average using the 'nonzero' function
y = [set(student_subject_matrix.applymap(lambda x: 1 if x >49.99 else 0).loc[i].iloc[ student_subject_matrix.applymap(
    lambda x: 1 if x >49.99 else 0).loc[i].to_numpy().nonzero()].index) for i in most_similar_students['Stuid']]
subj_passedby_mostsimstu= y[0].union(y[1],y[2],y[3],y[4],y[5],y[6],y[7],y[8],y[9])

# We have two sets of subjects taken and passed by "new_form3_student" and "most_similar_student(s)". 
# Set operation is used to find subjects taken by "most_similar_student(s)" but haven't been taken or passed by "new_form3_student"

# MAKING RECOMMENDATIONS
# A set of form 1 and 2 subjects
form1_2_subjects = ['ART_F1','COMP SC_F1','ELA_F1','FRE_F1','GEO_F1','HIS_F1','MTH_F1','RE_F1','SCI_F1','SH_F1','ART_F2',
                        'COMP SC_F2','ELA_F2','FRE_F2','GEO_F2','HIS_F2','MTH_F2','RE_F2','SCI_F2','SH_F2']
form1_2_subjects = set(form1_2_subjects)
    
# A set of core subject at form 3 and 4
core_subjects = ['MTH_F3&4','ELA_F3&4','RE_F3&4']
core_subjects = set(core_subjects)

# humanities and commercial subjects at form 3 and 4
hum_comm_subjects = ['ACC_F3&4','BST_F3&4','ART_F3&4','ELIT_F3&4','FRE_F3&4','GEO_F3&4','HIS_F3&4','PED_F3&4']
hum_comm_subjects = set(hum_comm_subjects)


# subjects to recommend to new form 3 student
subjects_to_recom = (subj_passedby_mostsimstu-subj_passedby_newf3stu-form1_2_subjects-core_subjects)


# subjects_to_recommend_to_new_form3_student
core = list(core_subjects)
for i in range(len(core_subjects)):
    if core[i]=='MTH_F3&4':
        core[i]='MATHEMATICS'
    elif core[i]=='ELA_F3&4':
        core[i]='ENGLISH LANGUAGE'
    elif core[i]=='RE_F3&4':
        core[i]='RELIGIOUS STUDIES'

recom = list(subjects_to_recom)
for i in range(len(subjects_to_recom)):
    if recom[i]=='ACC_F3&4':
        recom[i]='ACCOUNTING'
    elif recom[i]=='ART_F3&4':
        recom[i]='ART'
    elif recom[i]=='BIOL_F3&4':
        recom[i]='BIOLOGY'
    elif recom[i]=='BST_F3&4':
        recom[i]='BUSINESS STUDIES'
    elif recom[i]=='CHEM_F3&4':
        recom[i]='CHEMISTRY'
    elif recom[i]=='COMP SC_F3&4':
        recom[i]='COMPUTER SCIENCE'
    elif recom[i]=='D&T_F3&4':
        recom[i]='DESING AND TECHNOLOGY'
#    elif recom[i]=='ELA_F3&4':
#        recom[i]='ENGLISH LANGUAGE'
    elif recom[i]=='ELIT_F3&4':
        recom[i]='ENGLISH LITERATURE'
    elif recom[i]=='FRE_F3&4':
        recom[i]='FRENCH'
    elif recom[i]=='GEO_F3&4':
        recom[i]='GEOGRAPHY'
    elif recom[i]=='HIS_F3&4':
        recom[i]='HISTORY'
    elif recom[i]=='ICT_F3&4':
        recom[i]='INFORMATION COMMUNICATION TECHNOLOGY'
#    elif recom[i]=='MTH_F3&4':
#        recom[i]='MATHEMATICS'
    elif recom[i]=='PED_F3&4':
        recom[i]='PHYSICAL EDUCATION'
    elif recom[i]=='PHY_F3&4':
        recom[i]='PHYSICS'
#    elif recom[i]=='RE_F3&4':
#        recom[i]='RELIGIOUS STUDIES'

if st.button('Show Recommendations', type ='primary'):
    st.subheader("Recommended Subjects")
    st.write("========================================================================")
    if student_subject_matrix2.tail(1).loc[579].sum()==0:
        st.write("Please enter your marks on the subject marks entry form!!!!")
    elif student_subject_matrix2.tail(1).loc[579].sum()>0:
        st.subheader("Core Subjects")
        st.write(core)
        st.write(".........................................................................................................")
        st.subheader("Non-Core Subjects")
        st.write(recom)
        st.write("...........................................................................................................")


# function to reset the student_subject_matrix2
student_subject_matrix2=student_subject_matrix2.drop(student_subject_matrix2.index[-1], axis = 0)