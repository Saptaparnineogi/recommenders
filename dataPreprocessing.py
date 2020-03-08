# Create CSV files for base product rating and complementary product rating
# Create CSV files with base product and complementary product relation


import pandas as pd
import gzip
import numpy as np


# Read jason files for review and metadata
def parse(path):
    '''
    param path : path of the json file
    '''
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


# Create pandas dataframe from python dictionary
def getDF(path):
    '''
    param path : path of the json file
    '''
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


# Create rating matrix of all products for each user
def get_review(df1):
    '''
    param df1 : pandas dataframe with review data
    '''
    print("Creating review DF...")
    review_df = df1[["reviewerID", "asin", "overall"]]
    return review_df


# Preprocess meta-data dataframe
def get_metadata(df2):
    '''
    param df2 : pandas dataframe with meta-data
    '''
    print("Creating metadata DF...")
    df_related = df2["related"].apply(pd.Series)
    meta_df_without_related = pd.concat([df2, df_related], axis=1)\
        .drop(['related'], axis=1)
    return meta_df_without_related


# Create rating matrix of actual and complementary products for each user
def get_ratingmatrix(review_df, meta_df):
    '''
    param review_df : pandas dataframe with explicit rating information
    param meta_df : pandas dataframe with complementary product information
    '''
    print("Creating result DF...")
    comp_df = meta_df[['asin', 'also_bought']]
    comp_list = comp_df.apply(lambda x: pd.Series(x['also_bought']), axis=1)\
        .stack().reset_index(level=1, drop=True)
    comp_list.name = 'compProd'
    comp_df = comp_df.drop('also_bought', axis=1).join(comp_list)
    result = pd.merge(review_df, comp_df, on='asin')
    result = result.rename(columns={'overall': 'prodRating'})
    merged_df = pd\
        .merge(result, review_df, left_on=['compProd'], right_on=['asin'], how='left')\
        .drop(['reviewerID_y', 'asin_y'], axis=1)
    merged_df = merged_df.\
        rename(columns={'reviewerID_x': 'reviewerID', 'asin_x': 'productID', 'overall': 'compRating'})
    return merged_df


# Create CSV files with base products and complementary products
def create_actual_comp_prod(meta_df, category):
    '''
    param meta_df : pandas dataframe with complementary product information
    param category : string value, represents category of the products
    '''
    comp_df = meta_df[['asin', 'also_bought']]
    comp_list = comp_df.apply(lambda x: pd.Series(x['also_bought']), axis=1)\
        .stack().reset_index(level=1, drop=True)
    comp_list.name = 'compProd'
    comp_df = comp_df.drop('also_bought', axis=1).join(comp_list)
    comp_df.to_csv('asin_comp_{}.csv'.format(category), index=False)
    return comp_df


# Create rating file for base products only in CSV format
def get_mainprod_rating(review_df, asin_comp, category):
    '''
    param review_df : pandas dataframe with rating information
    param asin_comp : pandas dataframe with base and complementary product relationship
    param category : string value with category name
    '''
    comp_list = asin_comp['compProd'].dropna()
    main_rating_df = review_df[~review_df['asin'].isin(comp_list)]
    print("Main product rating shape:" + str(main_rating_df.shape))
    main_rating_df.to_csv('main_rating_df_{}.csv'.format(category), index=False)


# Create rating file for complementary products only in CSV format
def get_compprod_rating(review_df, asin_comp, category):
    '''
    param review_df : pandas dataframe with rating information
    param asin_comp : pandas dataframe with base and complementary product relationship
    param category : string value with category name
    '''
    comp_list = asin_comp['compProd'].dropna()
    comp_rating_df = review_df[review_df['asin'].isin(comp_list)]
    print("Comp rating shape:" + str(comp_rating_df.shape))
    comp_rating_df.to_csv('comp_rating_{}.csv'.format(category), index=False)


# Take input as json files and give output as csv files for base and
# complementary product rating and relationship
if __name__ == '__main__':
    category = 'Musical_Instruments'
    df1 = getDF('dataset/reviews_Musical_Instruments.json.gz')
    print("Shape of review dataframe:" + str(df1.shape))
    df2 = getDF('dataset/meta_Musical_Instruments.json.gz')
    print("Shape of metadata dataframe:" + str(df2.shape))
    review_df = get_review(df1)
    print(review_df.shape)
    meta_df = get_metadata(df2)
    asin_comp_df = create_actual_comp_prod(meta_df, category)
    get_mainprod_rating(review_df, asin_comp_df, category)
    get_compprod_rating(review_df, asin_comp_df, category)
