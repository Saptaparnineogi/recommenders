#!/usr/bin/env python
# coding: utf-8

# In[9]:


# Recommendation with custom function

from pyspark.sql import SparkSession
from array import array
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.functions import monotonically_increasing_id
from pandas import DataFrame
from pyspark.sql.window import Window
from pyspark.sql.functions import isnan
from pyspark.sql.types import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql import SQLContext
import pyspark.sql.functions as f
import random
import sys
from pyspark.sql.types import *
import numpy as np
from pyspark.sql.functions import size
import math


class ALSRecommender:
    global test

    def __init__(self, spark_session,
                 actual_prod_rating, comp_prod_rating, actual_comp):
        self.spark = spark_session
        self.sc = spark_session.sparkContext
        self.sqlContext = SQLContext(self.sc)
        self.actual_prod_df = self.load_file(actual_prod_rating)
        self.comp_prod_df = self.load_file(comp_prod_rating)
        self.asin_comp_df = self.load_file(actual_comp)
        (
                self.all_prod_df, self.actual_prod_with_all_ids,
                self.comp_prod_with_all_ids, self.user_hash_id_LUT,
                self.all_prod_id_LUT
        ) = self.dfs_for_als()
        self.comp_train, self.train, self.test = train_test_split
        (self.comp_prod_with_all_ids,
            self.actual_prod_with_all_ids)
        self.testInstance = getTestInstances(
                self.comp_prod_with_all_ids,
                self.train, self.test)
        self.avg_rating_df = self._avg_rating(self.train)
        self.target, self.negTarget = self.create_target_df()
        self.recommendations = self.create_model()

    def load_file(self, filename):
        """
        Load file for ALS model
        Parameters
        ----------
        filename: string, file to be loaded
        """
        df = self.spark.read.csv(filename, header=True)
        return df

    def dfs_for_als(self):
        """
        create dataframes with integer ids

        """
        comp_prod_count = self.comp_prod_df.groupby('reviewerID').count()
        comp_prod_count = comp_prod_count.filter(comp_prod_count['count'] >= 2)
        comp_prod_filtered = self.comp_prod_df.join(
                comp_prod_count, on='reviewerID').drop('count')
        prod_df = self.actual_prod_df.join(comp_prod_filtered, on='reviewerID')
        main_df = prod_df.select('reviewerID', 'productID', 'prodRating')
        main_df = main_df.dropDuplicates()
        comp_df = prod_df.select('reviewerID', 'compProd', 'compRating')
        comp_df = comp_df.dropDuplicates()
        all_prod_df = main_df.union(comp_df)
        all_prod_hashs = all_prod_df.select("productID").distinct()
        all_prod_id_LUT = self.sqlContext.createDataFrame(
                all_prod_hashs.rdd.map(lambda x: x[0]).zipWithIndex(),
                StructType([StructField("productID", StringType(), True),
                            StructField("productID_int", IntegerType(), True)])
                )
        user_hashs = self.actual_prod_df.select("reviewerID").distinct()
        user_hash_id_LUT = self.sqlContext.createDataFrame(
                user_hashs.rdd.map(lambda x: x[0]).zipWithIndex(),
                StructType([StructField("reviewerID", StringType(), True),
                            StructField("reviewerID_int", IntegerType(),
                                        True)]))
        actual_prod_with_user_ids = self.actual_prod_df.join(
                user_hash_id_LUT, self.actual_prod_df.reviewerID ==
                user_hash_id_LUT.reviewerID).
        select('reviewerID_int', 'productID', 'prodRating')
        actual_prod_with_all_ids = actual_prod_with_user_ids.join(
                all_prod_id_LUT, actual_prod_with_user_ids.productID ==
                all_prod_id_LUT.productID).
        select('reviewerID_int', 'productID_int', 'prodRating')
        comp_prod_with_user_ids = self.comp_prod_df.join(
                user_hash_id_LUT, self.comp_prod_df.reviewerID ==
                user_hash_id_LUT.reviewerID).
        select('reviewerID_int', 'compProd', 'compRating')
        comp_prod_with_all_ids = comp_prod_with_user_ids.join(
                all_prod_id_LUT, comp_prod_with_user_ids.compProd ==
                all_prod_id_LUT.productID).
        select('reviewerID_int', 'productID_int', 'compRating')
        comp_prod_with_all_ids = comp_prod_with_all_ids.dropDuplicates()
        return (all_prod_df,
                actual_prod_with_all_ids,
                comp_prod_with_all_ids,
                user_hash_id_LUT,
                all_prod_id_LUT)

    def create_model(self):
        """
        create prediction of rating for each product.
        Generate topK recommendations, return recommendations as a dataframe

        """
        topK = 5
        als = ALS
        (
                rank=12,
                maxIter=10,
                regParam=0.05,
                userCol="reviewerID_int",
                itemCol="productID_int",
                ratingCol="prodRating"
                )
        model = als.fit(self.train)
        predictions = model.transform(self.testInstance)
        predictions = predictions.na.drop()
        windowUserId = Window.partitionBy(predictions['reviewerID_int']).
        orderBy(predictions['prediction'].desc())
        recommendation_df = predictions.select(col('*'), row_number().
                                               over(windowUserId).
                                               alias("row_number")).
        where(col("row_number") <= topK)
        recommendation_df = recommendation_df.groupby("reviewerID_int").
        agg(f.collect_list("productID_int").alias("recommendation"))
        return recommendation_df

    def _avg_rating(self, train):
        """
        Calculate average ratings for each user.
        param train : dataframe for training
        
        """

        all_prod_with_rating = self.all_prod_df.filter(
                self.all_prod_df.prodRating != 0)
        avg_rating_df = all_prod_with_rating.groupby("reviewerID").
        agg(f.mean("prodRating").alias("avgRating"))
        avg_rating_df = avg_rating_df.join(
                self.user_hash_id_LUT, on='reviewerID')
        avg_rating_df = avg_rating_df.select('reviewerID_int', 'avgRating')
        avg_rating_df = avg_rating_df.dropDuplicates()
        return avg_rating_df

    def create_target_df(self):
        """
        Create dataframes with positively rated products and 
        negetively rated products
        
        """
        test_with_avg = self.test.join(
                self.avg_rating_df, "reviewerID_int")
        target_df = test_with_avg.filter
        (
                test_with_avg.prodRating >= test_with_avg.avgRating-2
                )
        neg_target_df = test_with_avg.filter
        (
                test_with_avg.prodRating < test_with_avg.avgRating-2
                )
        return (target_df,
                neg_target_df)



def chunktrain(library):
    """
    Create training data per user basis
    param library : list of rated products

    """
    random.shuffle(library)
    ln = len(library)
    a = int(0.8 * ln)
    lst1 = library[0:a]
    return lst1


def chunkTest(productlist, library):
    """
    create test data per user basis.
    param productlist : list of all rated products
    param library : list of products in training set

    """
        lst2 = [item for item in productlist if item not in library]
        return lst2


def train_test_split(comp_prod_with_all_ids, actual_prod_with_all_ids):
        """
        Create whole training and test set.
        training = 100% rating data for main products + 80% rating from comp prod
        test = 20% of rated products from comp prod.

        param comp_prod_with_all_ids : dataframe with comp prod rating
        param actual_prod_with_all_ids : dataframe with main prod rating
        
        """
        schema = ArrayType(StructType([
            StructField("productID_int", IntegerType(), False),
            StructField("compRating", IntegerType(), False)
            ]))
        comp_prod_with_all_ids = comp_prod_with_all_ids.withColumn
        (
                "compRating", comp_prod_with_all_ids["compRating"].
                cast(IntegerType()))
        comp_prod_with_all_ids = comp_prod_with_all_ids.withColumn
        (
                "NewColumn", struct(comp_prod_with_all_ids.productID_int,
                                    comp_prod_with_all_ids.compRating))
        comp_prod_with_all_ids = comp_prod_with_all_ids.groupby
        (
                "reviewerID_int").agg(f.collect_list("NewColumn")
                                      .alias("prod_rating_list"))
        SparseSplitLib = udf(chunktrain, schema)
        SparseSplitLibrary = udf(chunkTest, schema)
        comp_prod_with_all_ids_Train = comp_prod_with_all_ids.withColumn
        (
                "prod_rating_train", SparseSplitLib
                (
                    comp_prod_with_all_ids.prod_rating_list))
        comp_prod_with_all_ids_Test = comp_prod_with_all_ids_Train.withColumn
        ("prod_rating_test", SparseSplitLibrary("prod_rating_list",
                                                "prod_rating_train")).
        drop("prod_rating_list", "prod_rating_train")
        comp_prod_train = comp_prod_with_all_ids_Train.withColumn
        (
                "prod_rating_train", explode("prod_rating_train")).
        drop("prod_rating_list")
        comp_prod_test = comp_prod_with_all_ids_Test.withColumn
        (
                "prod_rating_test", explode("prod_rating_test")).
        drop("prod_rating_list")
        comp_prod_train = comp_prod_train.select
        (
                col("reviewerID_int"), col("prod_rating_train").
                productID_int.alias("productID_int"),
                col("prod_rating_train").compRating.alias("compRating"))
        comp_prod_test = comp_prod_test.select
        (
                col("reviewerID_int"), col("prod_rating_test").
                productID_int.alias("productID_int"),
                col("prod_rating_test").compRating.alias("compRating"))
        train = actual_prod_with_all_ids.union(comp_prod_train)
        train = train.withColumn
        (
                "prodRating", train["prodRating"].cast(IntegerType()))
        train = train.fillna(0)
        test = comp_prod_test.selectExpr(
                "reviewerID_int as reviewerID_int",
                "productID_int as productID_int",
                "compRating as prodRating").dropDuplicates()
        return (comp_prod_train, train, test)


def getTestNeg(test, train, all_comp):
        """
        SPARK UDF
        Create negative entries for the test set

        param test : dataframe with test data
        param train : dataframe with training data
        param all_comp : list with all comp prod ids
        
        """
        negNum = len(test)
        tmp_item = test
        for t in range(negNum):
            j = np.random.choice(all_comp)
            while j in train or j in tmp_item:
                j = np.random.choice(all_comp)
            tmp_item.append(j.item())
        return tmp_item


def getTestInstances(comp_df, train, test):
        """
        Return dataframe with whole test data
        using UDF
        
        """
        compprodlist = comp_df.select("productID_int").distinct()
        compprodlist = [int(row['productID_int']) for row in compprodlist.collect()]
        test_withoutrating = test.drop("prodRating")
        test_withoutrating = test_withoutrating.groupby("reviewerID_int").agg(f.collect_list("productID_int").alias("test_prod_list"))
        train_withoutrating = train.drop("prodRating")
        train_withoutrating = train_withoutrating.groupby("reviewerID_int").agg(f.collect_list("productID_int").alias("train_prod_list"))
        all_comp_df = test_withoutrating.join(train_withoutrating, "reviewerID_int")
        all_comp_df = all_comp_df.withColumn("comp_list", f.array([f.lit(x) for x in compprodlist]))
        udf_testNeg = udf(getTestNeg, ArrayType(IntegerType()))
        all_comp_df = all_comp_df.withColumn("whole_test", udf_testNeg(all_comp_df.test_prod_list, all_comp_df.train_prod_list, all_comp_df.comp_list))
        all_comp_df = all_comp_df.drop("test_prod_list", "train_prod_list", "comp_list")
        testInstance = all_comp_df.withColumn("productID_int", explode("whole_test"))
        testInstance = testInstance.drop("whole_test")
        return testInstance


def _create_hitlist(col1, col2):
    """
    SPARK UDF to calculate hit length
    param col1 : dataframe column with recommendation list
    param col2 : dataframe column with target list
    
    """

    if col1 is None:
        return 0.0
    else:
        if len(col1) < len(col2):
            hitList = [item for item in col1 if item in col2]
        else:
            hitList = [item for item in col2 if item in col1]
        return float(len(hitList))


def _precision(hitlen, recom):
    """
    SPARK UDF to calculate precision values
    param hitlen : integer hit length
    param recom : dataframe column with recommendations
    
    """
    topK = 5
    if recom is None:
        return 0.0
    else:
        return float(hitlen/topK)


def _recall(hitlen, relevant):
    """
    SPARK UDF to calculate recall values for each user
    param hitlen : integer hit length
    param relevant : dataframe column with relevant products
    
    """
    if relevant is None:
        return 0.0
    else:
        return float(hitlen/len(relevant))


def getIdealRecomList(ranklist, targetList, negTargetList):
    """
    SPARK UDF to get ideal recommendation list for each user
    param ranklist : test set list
    param targetList : relevant product list
    param negTargetList : negetive target list

    """
    idealRecomList = []
    topK = 5
    for i in ranklist:
        if i in targetList:
            idealRecomList.append(i)
    if negTargetList is None:
        for i in ranklist:
            if i not in targetList:
                idealRecomList.append(i)
    if negTargetList is not None:
        for i in ranklist:
            if i not in targetList and i not in negTargetList:
                idealRecomList.append(i)
        for i in ranklist:
            if i in negTargetList:
                idealRecomList.append(i)
    idealRecomList = idealRecomList[:topK]
    return idealRecomList


def getDCG(ranklist, targetItem):
    """
    SPARK UDF to calculate DCG value for each user
    param ranklist : recommendations / ideal recommendations
    param targetItem : relevant product list
    """
    dcg = 0.0
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item in targetItem:
            dcg += math.log(2) / math.log(i + 2)
    return float(dcg)


def getMRR(ranklist, targetItem):
    """
    SPARK UDF to calculate MRR value for each use.
    param ranklist : recommendations list
    param targetItem : relevant product list
    
    """
    mrr = 0.0
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item in targetItem:
            mrr = float(1 / (i + 1))
            return mrr
    return 0.0


def _evaluation(target, negTarget, recom_df, test):
    """
    Return evaluation metrics for whole dataset.
    param target : dataframe with relevant products
    param negTarget : dataframe with negetively rated products
    param recom_df : dataframe with recommendations
    param test : dataframe with test set

    """
    target_list = target.groupby("reviewerID_int").agg(f.collect_list("productID_int").alias("target"))
    evaluation_df = recom_df.join(target_list, "reviewerID_int")
    test = test.groupby('reviewerID_int').agg(f.collect_list("productID_int").alias("test"))
    evaluation_df = evaluation_df.join(test, 'reviewerID_int')
    negTarget_list = negTarget.groupby('reviewerID_int').agg(f.collect_list('productID_int').alias('negTarget'))
    evaluation_df = evaluation_df.join(negTarget_list, 'reviewerID_int', how='left')
    evaluation_df = evaluation_df.fillna(0)
    hitUdf = udf(_create_hitlist, FloatType())
    evaluation_df = evaluation_df.withColumn("hitLen", hitUdf("target", "recommendation"))
    zero_hit = evaluation_df.where(evaluation_df["hitLen"] == 0)
    zero_hit_count = zero_hit.count()
    precisionUdf = udf(_precision, FloatType())
    recallUdf = udf(_recall, FloatType())
    dcgUdf = udf(getDCG, FloatType())
    mrrUdf = udf(getMRR, FloatType())
    idealReecomUdf = udf(getIdealRecomList, ArrayType(IntegerType()))
    evaluation_df = evaluation_df.withColumn("idealRecom", idealReecomUdf("test", "target", "negTarget"))
    evaluation_df = evaluation_df.withColumn("precision", precisionUdf("hitLen", "recommendation"))
    evaluation_df = evaluation_df.withColumn("recall", recallUdf("hitLen", "target"))
    evaluation_df = evaluation_df.withColumn("dcg", dcgUdf("recommendation", "target"))
    evaluation_df = evaluation_df.withColumn("idcg", dcgUdf("idealRecom", "target"))
    evaluation_df = evaluation_df.withColumn("ndcg", evaluation_df.dcg/evaluation_df.idcg)
    evaluation_df = evaluation_df.withColumn("mrr", mrrUdf("recommendation", "target"))
    evaluation_result = evaluation_df.select(avg("precision").alias("precision"), avg("recall").alias("recall"), avg("ndcg").alias("NDCG"), avg("mrr").alias("mrr"))
    evaluation_result = evaluation_result.withColumn("zero_hit", lit(zero_hit_count))
    return evaluation_result


if __name__ == "__main__":
    spark = SparkSession.builder.appName("ALS recommender").getOrCreate()
    recommender = ALSRecommender(
        spark,
        "/user/neogis/main_rating_baby.csv",
        "/user/neogis/comp_rating_baby.csv",
        "/user/neogis/asin_comp_baby.csv")
    topK = 5
    target_count = recommender.target.groupby('reviewerID_int').count().alias("count")
    target_count = target_count.where(target_count['count'] >= topK)
    target_df = recommender.target.join(target_count, on='reviewerID_int')
    target_df = target_df.drop('count')
    evaluation_result = _evaluation(target_df, recommender.negTarget, recommender.recommendations, recommender.testInstance)
    print(evaluation_result.show())
