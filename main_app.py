#Library Imports
import streamlit as st

#Data Processing Libraries
import pandas as pd
import numpy as np

#Plotting Libraries
import seaborn as sns
from matplotlib import pyplot as plt
from PIL import Image

# Setting webpage Layout 
# st.set_page_config(layout = "wide") 

# Hide footer and Hamburger Menu
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Page Header
st.subheader("INLS625: INFORMATION ANALYTICS")
st.markdown("**Project: What's your personality? Let's Find Out..**")
st.write("*Vibhor Gupta*")
st.write("*2022-11-20*")

about, eda, dpm = st.tabs(["About", "Data Exploration", "Data Processing and Modelling	"])

with about:
	st.markdown("**The Project**")
	st.markdown("""<p align="justify">The project was the part of the course Information Analytics taught by Prof Rajasekar Arcot at UNC Chapel Hill.
		The goal of the project was to implement the learnings from the subject and develop something meainingful and novel using the knowledge acquired.
		For the purpose of the this project, I chose to identify the personality type of the users based on digital presence, which is discussed below.
		</p>""", unsafe_allow_html=True)
	st.markdown("**Personality**")
	st.markdown("""<p align = "justify">
		Personality is a set of charactersitcs, that defines a person, and these characterstics change depending on the circumstances.
		Infact, according to the internet, personality has it's root in the Latin word *persona*, which means mask.
		Research through out the time has been done, to understand people's personality. </p>
		""", unsafe_allow_html=True)
	st.markdown("""<p align = "justify">
		<b>The MBTI Personality:</b><br><br>
		The MBTI or Myers Briggs Type Indicator, was one of the way to classify personalities. The classification uses the following traits to identify personalties:
		</p>
		<ul>
		<li> Introvert or Extrovert </li>
		<li> Sensing or INtutive </li>
		<li> Thinking or Feeling </li>
		<li> Perceiving or Judging </li>
		</ul>
		<p align = "justify">
		The permution of the 8 types, gave rise two 16 different types as shown in the figure below:
		</p>
		<img src="https://upload.wikimedia.org/wikipedia/commons/1/1f/MyersBriggsTypes.png", height = 600, width = 850, align="center"></img>
		<figcaption style = "text-align: center;">The 16 Personality types </figcaption>
		<br>
		""", unsafe_allow_html=True)
	st.markdown("**Motivation**")
	st.markdown("""<p align = "justify">
		There were two motivations to pick up this  project :</p> 
		<ul>
		<li> The first was to understand how information is used to predict user behaviour and use the results to benefit big companies. </li>
		<li> The second was my curiousity to understand the different personality types that exist </li> 
		</ul>
		 """, unsafe_allow_html=True)
	st.markdown("**Data**")
	st.markdown("""<p align = "justify">
		On searching the web, I was able to find one of the <a href-"https://www.kaggle.com/datasets/datasnaek/mbti-type"> dataset </a> from Kaggle for the MBTI prediction.
		The dataset has comments from around than 8000 users and their personality type. I have used the dataset to predict personality types of users.
		 """, unsafe_allow_html=True)
	st.markdown("Goals")
	st.markdown("""<p align = "justify">
		The goals of the project were as follows </p>
		<ul>
		<li> To understand the information analytics process from data cleaning to generating insights </li>
		<li> To use nltk and text processing libraries in python </li>
		<ul>
		""", unsafe_allow_html=True)
	st.markdown("References:")
	st.markdown("""<ul>
		<li> Official MBTI <a href ="https://www.myersbriggs.org/my-mbti-personality-type/mbti-basics/">website</a></li>
		<li> 16 personalities <a href = "https://www.16personalities.com/free-personality-test">website</a> </li>
		<li> Kaggle data <a href="https://www.kaggle.com/datasets/datasnaek/mbti-type"> source </a> </li>
		<li> <a href ="https://doi.org/10.48550/arXiv.2201.08717">Publication</a> on MBTI prediction </li> 
		""", unsafe_allow_html=True)
	st.markdown("""
		**Note:**
		I have used python for the whole project and build and embeded the results in this website.
		 """, unsafe_allow_html=True)


with eda:
	st.markdown("**Exploratory Data Analysis**")
	st.markdown("*The dataset*")
	df = pd.read_csv("./Data/mbti_1.csv")
	st.dataframe(df.head())
	st.markdown("*Basic information about the data*")
	st.write(df.describe())
	st.markdown("**Inferences**")
	st.markdown("""
		<ul>
		<li> There are two columns as expected in the dataset </li>
		<li> The number of records in the dataset are 8675 with no null values</li>
		<li> Their are 16 type of personalities </li>
		<li> The posts columns ha 50 posts for each user separted by '|||' symbol </li>
		</ul>
		""",unsafe_allow_html=True)
	st.markdown('*Personality type distribution*')
	fig1, ax1 = plt.subplots()
	ax1 = sns.countplot(x=df["type"], order= df["type"].value_counts().index)
	plt.xlabel("Personality Type", fontsize = 12)
	plt.ylabel("Count of users", fontsize = 12)
	st.pyplot(fig1)
	st.markdown("**Inferences**")
	st.markdown('''
		<ul>
		<li> Clearly the distribution is skewed for the data, which needed to be handled </li> 
		</ul>
		''',unsafe_allow_html=True )
	st.markdown("*Let's see the WordCloud for the whole data*")
	img1 = Image.open("./Images/wordcloud.png")
	st.image(img1, caption = "WordCloud for the corpus")
	st.markdown("**Inferences**")
	st.markdown("""<ul>
		<li> In the 8675*50 = 433750 posts, I found many irrelevant words like V, P, Fe, jpg etc</li>
		<li> The word cloud was plotted for each of the personality type but due to space constraint they are not shown here </li>
		</ul>
		""",unsafe_allow_html=True )
	st.markdown("""*Few posts*""",unsafe_allow_html=True )
	st.write("Let's look at top 10 posts")
	def ind_post(record, posts):
		for post in record[1].split("|||"):
			posts.append((record[0],post))
		return posts

	posts = []
	df.apply(lambda x : ind_post(x, posts), axis=1)
	top10posts = pd.DataFrame(posts[0:10])
	top10posts.rename(columns={0:"type", 1:"Individual Posts"}, inplace= True)
	st.dataframe(top10posts)
	st.markdown("**Inferences**")
	st.markdown("""<ul>
		<li> The posts contain many urls, and they didn't help in personality classificaiton</li>
		<li> The words were in different cases, that were needed to be converted to the same case </li>
		<li> There were symbols, multiple punctuations etc, which were needed to be tackled as well </li>
		</ul>
		""",unsafe_allow_html=True )
	st.markdown(""" Looking at the data, things didn't seem straight forward and required text processing which is explained on the next tab.
				""")


with dpm:
	st.markdown("**Data Pre-Processing**")
	st.markdown("""<p align = "justify">
			As seen on the Data Exploration tab, the data had many issues starting with class imbalance and inconsistencies in the posts data.<br>
			To counter the post data inconsistencies, a pre processing pipeline was developed that took care of the urls, emojis, improper punctutaions etc.<br><br>
			The data was cleaned after passing thorough the pipeline, and the 10 posts from the data looked like :
			</p>""",
			unsafe_allow_html=True ) 
	preprocessed_df = pd.read_csv("./Data/preprocessed_data.csv")
	st.dataframe(preprocessed_df)
	st.markdown("**Modelling**")
	st.markdown("""<p align = "justify">
		The next steps followed for were as:
			</p>
			<ul>
			<li> Target ( type column in the data ) encoding to integer form </li>
			<li> Processing of posts using nltk and sklearn modules. Major steps involve : 
			<ul>
			<li>Removal of stop words from the posts </li>
			<li> Using Count verctorizer and TF-Idf vectorizer</li>
			</li>
			</ul>
			<li> Splitting the data into 70:30 train:test ratio </li>
			<li>Four models were trained on the data which were:
			<ul>
			<li> Decision Tree </li>
			<li> Logistic Regression </li>	
			<li> Random Forest </li>
			<li> SVM </li>
			</ul></li>
			<li> Model evaluation by comparing test accuracies</li>
			<ul>
			""",unsafe_allow_html=True )
	st.markdown("""<p align = "justify">
		The final results from the models evaluation were shown in the following figure: </p>
			""",unsafe_allow_html=True )
	img2 = Image.open("./Images/accuracy%.png")
	st.image(img2, caption ="Test accuracy (%) for the models")
	st.markdown("**Inferences**")
	st.markdown("""
		<ul>
		<li>The test accuracies from the models were able to raeach at max 50% for Logistic Regression and SVM</li>
		<li> The accuracies are too less, to deploy the model</li>
		<li> Playing with the model hyperparameters may increase the accuracy a litle bit </li>
		<li> The major fall back on the accuracy is due to the data, either more data or feautres  are needed to be build to improve on accuracy </li>
		</ul>
		""",unsafe_allow_html=True)
	st.subheader("Project Learnings")
	st.markdown("""
		<ul>
		<li>The project gave me a hands on to tackle a data science project from data collection to making insights</li>
		<li> I was able to learn text processing libraries, like nltk and different modeling techinques from sklearn </li>
		<li> Debugging the model proposes was another challenge, looking at the code and the model performance</li>
		<li> This was a fun project, with many things to learn, although the model didn;t come out well but still I enjoyed the experience I got in the project</li>
		</ul>
		""",unsafe_allow_html=True)

	st.markdown("**Note:** The code for the programs is either running on the backend of the website or is available on GitHub")