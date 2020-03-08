#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import math


class itemPopularity():
    def __init__(self):
        self.topK = 5
        self.mainProdData = 'generatedData/actual_rating.csv'
        self.compProdData = 'generatedData/comp_rating.csv'
        self.mainProdCompProd = 'generatedData/asin_comp.csv'
        self.main_rev, self.comp_rev, self.asin_comp = self.creatDf()
        self.createModel(self.main_rev, self.comp_rev, self.asin_comp)

    def creatDf(self):
        main_rev = pd.read_csv(self.mainProdData)
        comp_rev = pd.read_csv(self.compProdData)
        asin_comp = pd.read_csv(self.mainProdCompProd)
        return main_rev, comp_rev, asin_comp

    def createModel(self, mainProd, compProd, mainComp):
        mainProd = mainProd.dropna()
        compProd = compProd.dropna()
        mainComp = mainComp.dropna()
        compProd_grouped = compProd.groupby('compProd').agg({'reviewerID': 'count'}).reset_index()
        compProd_grouped.rename(columns={'reviewerID': 'score'},inplace=True)
        main_comp_score = mainComp.merge(compProd_grouped, on='compProd', how='left')
        main_comp_score = main_comp_score.fillna(0.0)
        main_comp_score = main_comp_score.groupby(["asin"]).apply(lambda x: x.sort_values(["score"], ascending=False)).reset_index(drop=True)
        main_comp_topK = main_comp_score.groupby('asin', as_index=False).head(self.topK)
        main_comp_rank = main_comp_topK.groupby('asin', as_index=False).agg(list)
        user_comp_rank = mainProd.merge(main_comp_rank, left_on='productID', right_on='asin')
        self.user_grouped = user_comp_rank.groupby('reviewerID', as_index=False).agg(list)
        self.user_grouped = self.user_grouped.drop(columns=['prodRating', 'asin'])

    def recommend(self, compProd, topK):
        if len(compProd) > 1:
            reqProd = int(topK / len(compProd))
        else:
            reqProd = topK
        recom_list = []
        for l in compProd:
            if len(l) > reqProd:
                for i in range(0, reqProd):
                    recom_list.append(l[i])
            else:
                for i in range(0, len(l)):
                    recom_list.append(l[i])
        return recom_list

    def maketargetSet(self):
        avg_rate_df = self.comp_rev.groupby('reviewerID', as_index=False).mean()
        avg_rate_df = avg_rate_df.rename(columns={'compRating': 'avgRating'})
        comp_with_avg = self.comp_rev.merge(avg_rate_df, on="reviewerID")
        target = comp_with_avg.loc[comp_with_avg['compRating'] >= comp_with_avg['avgRating']]
        target = target.drop(columns=['compRating', 'avgRating'])
        negTarget = comp_with_avg.loc[comp_with_avg['compRating'] < comp_with_avg['avgRating']]
        negTarget = negTarget.drop(columns=['compRating', 'avgRating'])
        return target, negTarget

    def makeRecommendation(self):
        self.user_grouped['recommendation'] = self.user_grouped.apply(lambda x: self.recommend(x['compProd'], self.topK), axis=1)
        recom_df = self.user_grouped[['reviewerID', 'recommendation']]
        return recom_df


def evaluate(recom_df, targetTest, negTargetTest, topK):
    target_df = targetTest.groupby('reviewerID', as_index=False).agg(list)
    neg_target_df = negTargetTest.groupby('reviewerID', as_index=False).agg(list)
    recom_with_target = recom_df.merge(target_df, on='reviewerID')
    recom_with_target = recom_with_target.rename(columns={'compProd': 'target'})
    recom_with_target = recom_with_target.merge(neg_target_df, on='reviewerID', how='left')
    recom_with_target = recom_with_target.rename(columns={'compProd': 'negTarget'})
    recom_with_target['negTarget'] = recom_with_target['negTarget'].fillna(value=0.0)
    recom_with_target['idealRecomList'] = recom_with_target.apply(lambda x: getIdealRecomList(x['recommendation'], x['target'], x['negTarget']), axis=1)
    recom_with_target['hitLen'] = recom_with_target.apply(lambda x: getHit(x['recommendation'], x['target']), axis=1)
    recom_with_target['precision'] = recom_with_target.apply(lambda x: getPrecision(x['hitLen'], topK), axis=1)
    recom_with_target['recall'] = recom_with_target.apply(lambda x: getRecall(x['hitLen'], x['target']), axis=1)
    recom_with_target['dcg'] = recom_with_target.apply(lambda x: getDCG(x['recommendation'], x['target']), axis=1)
    recom_with_target['idcg'] = recom_with_target.apply(lambda x: getDCG(x['idealRecomList'], x['target']), axis=1)
    recom_with_target['ndcg'] = recom_with_target.apply(lambda x: getNDCG(x['dcg'], x['idcg']), axis=1)
    count = 0
    for i, row in recom_with_target.iterrows():
        if row['hitLen'] == 0:
            count += 1
    print("Nomer of zero hit : " + str(count))
    print("Precision@{} : ".format(topK) + str(recom_with_target['precision'].mean()))
    print("Recall@{} : ".format(topK) + str(recom_with_target['recall'].mean()))
    print("NDCG :" + str(recom_with_target['ndcg'].mean()))


def getHit(ranklist, targetItem):
    count = 0
    for item in ranklist:
        if item in targetItem:
            count += 1
    return count


def getPrecision(hit, k):
    precision_ = float(hit) / float(k)
    return precision_


def getRecall(hit, target):
    recall = float(hit) / float(len(target))
    return recall


def getDCG(ranklist, targetItem):
    dcg = 0
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item in targetItem:
            dcg += math.log(2) / math.log(i+2)
    return dcg


def getNDCG(dcg, idcg):
    if idcg != 0.0:
        return dcg / idcg
    else:
        return 0.0


def getIdealRecomList(ranklist, targetList, negTargetList):
    idealRecomList = []
    for i in ranklist:
        if i in targetList:
            idealRecomList.append(i)
    if negTargetList != 0.0:
        for i in ranklist:
            if i not in targetList and i not in negTargetList:
                idealRecomList.append(i)
        for i in ranklist:
            if i in negTargetList:
                idealRecomList.append(i)
    else:
        for i in ranklist:
            if i not in targetList:
                idealRecomList.append(i)
    return idealRecomList


if __name__ == '__main__':
    ip = itemPopularity()
    recommendation = ip.makeRecommendation()
    topK = ip.topK
    posTarget, negTarget = ip.maketargetSet()
    evaluate(recommendation, posTarget, negTarget, topK)
