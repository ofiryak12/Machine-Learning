from LinearRegression import Linear_Regression

linear = Linear_Regression('Salary_Data.csv',0,1) # 1.DataSet 2. Charicharistics column 3. variables column
print(linear.predict(3)) # predict outcome for a specific value
linear.test_visulazation() # Visualize the model