import streamlit as st
import time

st.sidebar.radio("Check this out", ["1", "2"])

st.write(time.time())

test_model_help = """
After training on the training data and validating on the validation data, we test the final prediction power of our model by running it on the test dataset that the algorithm has NEVER seen before.

It is very important to realize that fiddling with the hyperparameters overfits the validation dataset.
    
The test is the absolute final instance. You should not test before you are completely done with adjusting your model. 
    
If you adjust your model after testing, you will start overfitting the test dataset, which will defeat its purpose.
"""

if st.button("Test The Model"):
    col_1, col_2 = st.beta_columns(2,1)
    col_1.write(test_model_help)
    col_2.image("https://pbs.twimg.com/media/ESY0WNGU4AA3P0S.jpg")
    if st.button("Roger that.. run the test"):
        st.write("Yes")