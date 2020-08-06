#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator , VectorIndexer,VectorAssembler,IndexToString
from pyspark.ml.linalg import Vector
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics

import os
import numpy as np
from datetime import datetime
import array
import matplotlib.pyplot as plt


# In[2]:


start = datetime.now()


# In[3]:


spark = SparkSession.builder.appName("SQLFinalProject").config("spark.master", "local").getOrCreate()


# In[4]:


folder_path="E:\\BigDataProject\\brazilian-ecommerce"
customers =spark.read.csv(os.path.join(folder_path,"olist_customers_dataset.csv"),header=True)
orderItems = spark.read.csv(os.path.join(folder_path,"olist_order_items_dataset.csv"),header=True)
orderPayments = spark.read.csv(os.path.join(folder_path,"olist_order_payments_dataset.csv"),header=True)
orderReviews = spark.read.csv(os.path.join(folder_path,"olist_order_reviews_dataset.csv"),header=True)
orders = spark.read.csv(os.path.join(folder_path,"olist_orders_dataset.csv"),header=True)
products = spark.read.csv(os.path.join(folder_path,"olist_products_dataset.csv"),header=True)
sellers = spark.read.csv(os.path.join(folder_path,"olist_sellers_dataset.csv"),header=True)


# In[5]:


#customers.printSchema();


# # number of customers and sellers in each city

# get number of orders for each customer 

# In[6]:


customersOrders=customers.join(orders,["customer_id"],"inner").drop("order_purchase_timestamp","order_approved_at","order_delivered_carrier_date"
                ,"order_delivered_customer_date","order_estimated_delivery_date")
#print("customersOrders rows number: ",customersOrders.count());


cutomersOrders_OrdersCount= customersOrders.groupBy(col("customer_unique_id")).agg(count("customer_id"                        )).withColumnRenamed("count(customer_id)","orders_count").join(customersOrders,"customer_unique_id")
cutomersOrders_OrdersCount=cutomersOrders_OrdersCount.drop("customer_zip_code_prefix","customer_state","order_status")

"""
#show number of orders for each customer
print("cutomersOrders_OrdersCount rows count: ",cutomersOrders_OrdersCount.count())
cutomersOrders_OrdersCount.printSchema()
cutomersOrders_OrdersCount.sort().show()
#cutomersOrders_OrdersCount.printSchema()
"""


# ### Number of customers in each city

# In[7]:



customersOrders_OrdersCount_cityCount = cutomersOrders_OrdersCount.groupBy(col("customer_city"        )).agg(count("customer_unique_id")).withColumnRenamed("count(customer_unique_id)","customer_city_count"        ).join(cutomersOrders_OrdersCount,"customer_city");
print("cutomersOrders_OrdersCount_cityCount rows count: ",customersOrders_OrdersCount_cityCount.count());
#customersOrders_OrdersCount_cityCount.printSchema();
#Show # of customers in each city
customersOrders_OrdersCount_cityCountShow= customersOrders_OrdersCount_cityCount.select(                 "customer_city","customer_city_count").distinct().sort(col("customer_city_count").desc())                                .limit(20);
#save number of users in each city
#customersOrders_OrdersCount_cityCountShow.coalesce(1).write.format("csv").option("header",True).save("E:/BigDataProject/MyTables/customersOrders_OrdersCount_cityCount_python.csv");
customersOrders_OrdersCount_cityCountShow.show();


"""
#Show highest 20 customer cities
#draw the result
y=np.asarray(customersOrders_OrdersCount_cityCountShow.select("customer_city_count").collect()).flatten()
x=np.asarray(customersOrders_OrdersCount_cityCountShow.select("customer_city").collect()).flatten()


plt.bar(x ,y)
plt.title("Customer in each city")
plt.ylabel("Customer count")
plt.xlabel("Customer City")
plt.show()"""


# ### Number of seller in each city 

# In[8]:


#show # of sellers for each city
SellerCities=sellers.groupBy(col("seller_city")).agg(count("seller_id")).withColumnRenamed("count(seller_id)","seller_city_count");
print("SellerCities rows count: ",SellerCities.count());
#SellerCities.printSchema()
SellerCitiesShow=SellerCities.select("seller_city","seller_city_count").distinct().sort(col("seller_city_count").desc()).limit(20);
#save number of sellers in each city
#SellerCitiesShow.coalesce(1).write.format("csv").option("header",True).save("E:/BigDataProject/MyTables/sellersCities_python.csv");
SellerCitiesShow.show();
"""
y=np.asarray(SellerCitiesShow.select("seller_city_count").collect()).flatten()
x=np.asarray(SellerCitiesShow.select("seller_city").collect()).flatten()
plt.bar(x ,y)
plt.title("Seller in each city")
plt.ylabel("Seller count")
plt.xlabel("Seller City")
plt.show()"""


# ## number of orders from each category in each city

# In[9]:


customerId=customers.select("customer_id","customer_city")
orders_customerId=orders.join(customerId,["customer_id"],"inner").drop("order_status",                            "order_purchase_timestamp","order_approved_at","order_delivered_carrier_date"                            ,"order_delivered_customer_date","order_estimated_delivery_date")
#print(orders_customerId.count())
#orders_customerId.printSchema()

orders_customerId_orderItems=orders_customerId.join(orderItems,["order_id"],"inner").drop(                                               "shipping_limit_date","price","freight_value")
#print(orders_customerId_orderItems.count())
#orders_customerId_orderItems.printSchema()
#print(orders_customerId_orderItems.select("order_id").distinct().count())

orders_customerId_orderItems_products=orders_customerId_orderItems.join(products,["product_id"],"inner").drop(                                    "product_name_lenght","product_description_lenght","product_photos_qty",                                    "product_weight_g","product_length_cm","product_height_cm","product_width_cm")
#print(orders_customerId_orderItems_products.count())
#orders_customerId_orderItems_products.printSchema()
#orders_customerId_orderItems_products.show()

orders_customerId_orderItems_products=orders_customerId_orderItems.join(products,["product_id"],"inner").drop("product_name_lenght"                                    ,"product_description_lenght","product_photos_qty","product_weight_g","product_length_cm"                                    ,"product_height_cm","product_width_cm")
#print(orders_customerId_orderItems_products.count())
#orders_customerId_orderItems_products.printSchema()
#print(orders_customerId_orderItems_products.select("order_id").distinct().count())
#orders_customerId_orderItems_products.show()

o=orders_customerId_orderItems_products.select("customer_city","product_category_name")
category= o.select("product_category_name").distinct()
print(category.count())
print("schema of table to count number of orders of each category from each city")
orders_customerId_orderItems_products.printSchema()
#print("number of records",orders_customerId_orderItems_products.count())


product_category_name_orderId=orders_customerId_orderItems_products.groupBy("customer_city","product_category_name").agg(count("product_category_name"),                                                                                            count("customer_city")).drop("count(customer_city)")


product_category_name_orderId=product_category_name_orderId.sort(col("count(product_category_name)").desc())
#product_category_name_orderId_show=product_category_name_orderId.filter("product_category_name= 'cama_mesa_banho'").coalesce(1).write.format("csv"\
 #                                 ).option("header",True).save("E:/BigDataProject/MyTables/category_customer_python.csv");
product_category_name_orderId.show(50)
print("----------------------------------------\n")
product_category_name_orderId_show=product_category_name_orderId.filter("product_category_name= 'cama_mesa_banho'").limit(20)
print("number of sellers for cama_mesa_banho category ")
product_category_name_orderId_show.show()
"""
#show number of sellers for cama_mesa_banho category
y=np.asarray(product_category_name_orderId_show.select("count(product_category_name)").collect()).flatten()
x=np.asarray(product_category_name_orderId_show.select("customer_city").collect()).flatten()
plt.bar(x ,y)
plt.title("ordered product of category from each city")
plt.ylabel("Seller count")
plt.xlabel("Seller City")
plt.show()                                                                                                         
"""                                                                                                 


# In[10]:


#product_category_name_orderId.filter("customer_city = 'sao paulo'").coalesce(1).write.format("csv").option("header",True).save("E:/BigDataProject/MyTables/category_customer_sao_paulo.csv");


# ## sao paulo analysis

# ### category does not have seller in sao paulo

# get number of sellers for each category

# In[11]:


p=products.select('product_id','product_category_name')
result=orderItems.join(sellers,['seller_id'],'inner').drop('shipping_limit_date',                    'price','freight_value','seller_zip_code_prefix','seller_zip_code_prefix'                    ).join(p,['product_id'],'inner').filter("seller_city ='sao paulo'")


result=result.select("seller_id","seller_city","product_category_name").distinct().groupBy(                    "product_category_name","seller_city").agg(count("seller_id").alias("sellers_number"),count(                     "seller_id")).drop("count(seller_id)")
result.sort(col("sellers_number").desc()).show()
#result.groupBy("seller_city").agg(sum("sellers_number")).show()
#result.show()


# categories doesn't have seller in sao paulo

# In[12]:


#filter the sao paulo sellers
sao_paulo_sellers=sellers.filter("seller_city = 'sao paulo'")
#orders_customerId_orderItems_products.printSchema()


#know the categories that have sellers in sao paul sell them
sao_paulo=sao_paulo_sellers.join(orders_customerId_orderItems_products,["seller_id"],"inner")

sao_paulo=sao_paulo.select("seller_id","seller_city","product_category_name").distinct()
#sao_paulo.printSchema()
sao_paulo.show()

#show number of sellers for each category in sao paulo
sao_paulo_groupBy=sao_paulo.groupBy("product_category_name","seller_city").agg(count("seller_id").alias("sellers_count"),count("seller_city")).drop("count(seller_city)")
sao_paulo_groupBy=sao_paulo_groupBy.sort(col("sellers_count").desc())
sao_paulo_groupBy.show()

#how many sellers for all category
sao_paulo_groupBy.groupBy("seller_city").agg(sum("sellers_count")).show()

#right anit join
#categories doesn't have sellers in sao paulo 
c=sao_paulo_groupBy.join(category,["product_category_name"],"rightOuter").filter("seller_city is null AND product_category_name is not null")

#show the categoeies doesn't have sellers in sao paulo
print("category which has no sellers in sao paulo")
c.select("product_category_name").show()

#sao_paulo_groupBy.coalesce(1).write.format("csv").option("header",True).save("E:/BigDataProject/MyTables/sao_pa.csv");


# # Preferred payment way

# In[13]:


#because the customer can pay by more than one method
order_payments_orderItems=orderPayments.drop("payment_sequential","payment_installments").join(orderItems,"order_id").drop("shipping_limit_date","price","freight_value")
payment_type=order_payments_orderItems.groupBy("payment_type").agg(count("order_id").alias("count_of_orders"))
payment_type = payment_type.sort(col("count_of_orders").desc())
payment_type.show()
#saveing the sresults
#payment_type.coalesce(1).write.format("csv").option("header",True).save("E:/BigDataProject/MyTables/payment_type.csv");
"""
#draw the result
y=np.asarray(payment_type.select("count_of_orders").collect()).flatten()
x=np.asarray(payment_type.select("payment_type").collect()).flatten()
#it doesn't matter the payement value,the users prefer credit_card to pay
print(type(y[0]))
type(x)
plt.bar(x ,y)
plt.title("Payment way")
plt.ylabel("Payment number")
plt.xlabel("Payment way")
plt.show()"""


# # Orders Reviews select the category that get highest 5 review

# In[14]:


c=customers.select("customer_id","customer_unique_id","customer_city")
o=orders.select("order_id","customer_id")
oI=orderItems.select("order_id","product_id")
p=products.select("product_id","product_category_name")
oR=orderReviews.select("review_id","order_id","review_score").join(oI,"order_id").join(p,"product_id")
oR=oR.na.drop().distinct()
o=oR.groupBy("product_category_name" ,"review_score" ).agg(count("review_score").alias("count_of_reviews"),count("order_id")).drop("count(order_id)")
o.sort(col("review_score").desc(),col("count_of_reviews").desc()).show()
#.sort(col("review_score").desc(),col("count_of_reviews").desc()).limit(30).coalesce(1).write.format("csv").option("header",True).save("E:/BigDataProject/MyTables/score_reviews")


# 
# # Model to predict product category

# In[15]:


spark = SparkSession.builder.appName("SQLFinalProject").config("spark.master", "local").getOrCreate()


# In[16]:


p=products.groupBy("product_category_name").agg(count("product_id").alias("products_count"))

#print(classes_names)
#print("number of classes: ",p.count())
p=p.sort(col("products_count").desc())
p.show()

p=p.filter(col("products_count")>=2000)
classes_names=np.asarray(p.select("product_category_name").collect())

data=p.join(products,["product_category_name"],"inner").drop("products_count","product_id")
data.dtypes


# In[17]:


#drop row containing null
data=data.na.drop()
data.count()


# In[18]:


#preparing dataset
categoricalColumns = ['product_name_lenght','product_description_lenght','product_photos_qty','product_weight_g','product_length_cm','product_height_cm','product_width_cm']
stages = []
for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]
label_stringIdx = StringIndexer(inputCol = 'product_category_name', outputCol = 'label')
stages += [label_stringIdx]
assemblerInputs = [c + "classVec" for c in categoricalColumns] 
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]
print(stages)


# In[19]:


#use pipeline
pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(data)
cols = data.columns
df = pipelineModel.transform(data)
selectedCols = ['label', 'features'] + cols
df = df.select(selectedCols)
df.printSchema()


# ### Divide the total dataset into 9/1

# In[20]:


train, test = df.randomSplit([0.9, 0.1], seed = 2018)
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))


# ### train and test the model

# In[21]:


train.cache()
test.cache()

lr = LogisticRegression(maxIter=60, regParam=0.1)
lrModel = lr.fit(train)
result=lrModel.transform(test)
predictionAndLabels=result.select("label","prediction").rdd


# ### evaluate the training 

# In[22]:


trainingSummary = lrModel.summary

# Obtain the objective per iteration "error in each iteration"
objectiveHistory = trainingSummary.objectiveHistory

plt.plot(objectiveHistory)
plt.ylabel('Error')
plt.xlabel('Iteration')
plt.title("Training Error")
plt.show()
"""
#print("objectiveHistory:")
for objective in objectiveHistory:
    print(objective)
"""


# In[25]:


# for multiclass, we can inspect metrics on a per-label basis
"""
print("False positive rate by label:")
i_arr=[i for i in range(len(classes_names))]
f_p_rate_arr=[rate*100 for  rate in trainingSummary.falsePositiveRateByLabel]

for i, rate in enumerate(trainingSummary.falsePositiveRateByLabel):
    print("label %d: %s" % (i, rate))

plt.bar(i_arr ,f_p_rate_arr)
plt.title("False positive rate by label")
plt.ylabel("False positive")
plt.xlabel("Classes")
plt.show()


print("True positive rate by label:")
t_p_rate_arr=[rate*100 for  rate in trainingSummary.truePositiveRateByLabel]


plt.bar(i_arr ,t_p_rate_arr)
plt.title("True positive rate by label")
plt.ylabel("True positive")
plt.xlabel("Classes")
plt.show()


for i, rate in enumerate(trainingSummary.truePositiveRateByLabel):
    print("label %d: %s" % (i, rate))
    
"""

print("Precision by label:")
i_arr=[i for i in range(len(classes_names))]
p_arr=[rate*100 for  rate in trainingSummary.precisionByLabel]
plt.bar(i_arr ,p_arr)
plt.title("precision rate by label")
plt.ylabel("precision")
plt.xlabel("Classes")
plt.show()

"""
for i, prec in enumerate(trainingSummary.precisionByLabel):
    print("label %d: %s" % (i, prec))
"""
print("Recall by label:")
r_arr=[rate*100 for  rate in trainingSummary.recallByLabel]
plt.bar(i_arr ,r_arr)
plt.title("recall by label")
plt.ylabel("recall")
plt.xlabel("Classes")
plt.show()

"""
for i, rec in enumerate(trainingSummary.recallByLabel):
    print("label %d: %s" % (i, rec))
"""
print("F-measure by label:")
f_arr=[rate*100 for  rate in trainingSummary.fMeasureByLabel()]
plt.bar(i_arr ,f_arr)
plt.title("fMeasure by label")
plt.ylabel("fMeasureByLabel")
plt.xlabel("Classes")
plt.show()
"""
for i, f in enumerate(trainingSummary.fMeasureByLabel()):
    print("label %d: %s" % (i, f))
"""    
    
    
    
accuracy = trainingSummary.accuracy
fMeasure = trainingSummary.weightedFMeasure()
precision = trainingSummary.weightedPrecision
recall = trainingSummary.weightedRecall
print("Accuracy: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
      % (accuracy, fMeasure, precision, recall))


# ### evaluate the trained model during test

# In[26]:



#predictionAndLabels = test.map(lambda lp: (float(lrModel.predict(lp.features)), lp.label))
metrics = MulticlassMetrics(predictionAndLabels)
ac=metrics.accuracy
precision = metrics.precision()
recall = metrics.recall()
f1Score = metrics.fMeasure()
print("Summary Stats")
print("Precision = ", precision)
print("Recall = ", recall)
print("F Meaure = ", f1Score)
print("Accuracy = ",ac)


# In[27]:



converter = IndexToString(inputCol="label", outputCol="originalCategory")
converted = converter.transform(result)

converted.select("label","originalCategory").distinct().sort(col("label").asc()).show()


# In[28]:


end = datetime.now()
run_time=end-start
print("Total run time: ",run_time)


# In[ ]:




