from bdb import set_trace
import pandas as pd
import numpy as np
from olist.utils import haversine_distance
from olist.data import Olist


class Order:
    '''
    DataFrames containing all orders as index,
    and various properties of these orders as columns
    '''
    def __init__(self):
        # Assign an attribute ".data" to all new instances of Order
        self.data = Olist().get_data()

    def get_wait_time(self, is_delivered=True):
        """
        Returns a DataFrame with:
        [order_id, wait_time, expected_wait_time, delay_vs_expected, order_status]
        and filters out non-delivered orders unless specified
        """
        # Hint: Within this instance method, you have access to the instance of the class Order in the variable self, as well as all its attributes
        orders = self.data['orders'].copy()
        # filter on delivered orders
        if is_delivered:
            orders = orders.query("order_status == 'delivered'")
        # convert to datetime
        for i in range(3,8):
            orders.iloc[:,i] = pd.to_datetime(orders.iloc[:,i])
        # Compute wait_time
        orders.loc[:,'wait_time'] = \
            (orders.order_delivered_customer_date - orders.order_purchase_timestamp).dt.days *1.0
        # Compute expected_wait_time
        orders.loc[:,'expected_wait_time'] = \
            (orders.order_estimated_delivery_date - orders.order_purchase_timestamp).dt.days * 1.0
        # Compute delay_vs_expected
        orders.loc[:,'delay_vs_expected'] = \
            (orders.wait_time - orders.expected_wait_time).apply(lambda a: max(a,0))
        # filter only required columns, delete na
        return orders[['order_id', 'wait_time', 'expected_wait_time', 'delay_vs_expected', 'order_status']]

    def get_review_score(self):
        """
        Returns a DataFrame with:
        order_id, dim_is_five_star, dim_is_one_star, review_score
        """
        reviews = self.data['order_reviews'].copy()

        def dim_five_star(d):
            if d == 5:
                return 1
            else:
                return 0

        def dim_one_star(d):
            if d == 1:
                return 1
            else:
                return 0

        # dim_is_five_star
        reviews.loc[:,'dim_is_five_star'] = \
            reviews['review_score'].apply(dim_five_star)
        # dim_is_one_star
        reviews.loc[:,'dim_is_one_star'] = \
            reviews['review_score'].apply(dim_one_star)
        # correct order
        return reviews[[
            'order_id', 'dim_is_five_star', 'dim_is_one_star', 'review_score'
        ]]

    def get_number_products(self):
        """
        Returns a DataFrame with:
        order_id, number_of_products
        """
        # get order_items dataset
        items = self.data['order_items'][['order_id', 'product_id']]
        # groupy order_id, sum product_id
        items = items.groupby('order_id', as_index=False).count()
        items = items.rename(columns = {'product_id':'number_of_products'})
        return items

    def get_number_sellers(self):
        """
        Returns a DataFrame with:
        order_id, number_of_sellers
        """
        # get data
        sellers = self.data['order_items'][['order_id', 'seller_id']]
        # groupby order_id, count unique sellers per order (some products have the same seller)
        sellers = sellers.groupby('order_id').nunique()
        # rename
        sellers = sellers.rename(columns = {'seller_id':'number_of_sellers'})
        # reset index
        sellers = sellers.reset_index()
        return sellers

    def get_price_and_freight(self):
        """
        Returns a DataFrame with:
        order_id, price, freight_value
        """
        # get data
        pricefreight = self.data['order_items'][['order_id', 'price', 'freight_value']]
        # groupby, sum, reset_index
        pricefreight = pricefreight.groupby('order_id').sum().reset_index()
        return pricefreight

    # Optional
    def get_distance_seller_customer(self):
        """
        Returns a DataFrame with:
        order_id, distance_seller_customer
        """
        # import data
        data = self.data
        orders = data['orders']
        order_items = data['order_items']
        sellers = data['sellers']
        customers = data['customers']

        # Since one zip code can map to multiple (lat, lng), take the first one
        geo = data['geolocation']
        geo = geo.groupby('geolocation_zip_code_prefix',
                          as_index=False).first()

        # Merge geo_location for sellers
        sellers_mask_columns = [
            'seller_id', 'seller_zip_code_prefix', 'geolocation_lat', 'geolocation_lng'
        ]

        sellers_geo = sellers.merge(
            geo,
            how='left',
            left_on='seller_zip_code_prefix',
            right_on='geolocation_zip_code_prefix')[sellers_mask_columns]

        # Merge geo_location for customers
        customers_mask_columns = ['customer_id', 'customer_zip_code_prefix', 'geolocation_lat', 'geolocation_lng']

        customers_geo = customers.merge(
            geo,
            how='left',
            left_on='customer_zip_code_prefix',
            right_on='geolocation_zip_code_prefix')[customers_mask_columns]

        # Match customers with sellers in one table
        customers_sellers = customers.merge(orders, on='customer_id')\
            .merge(order_items, on='order_id')\
            .merge(sellers, on='seller_id')\
            [['order_id', 'customer_id','customer_zip_code_prefix', 'seller_id', 'seller_zip_code_prefix']]

        # Add the geoloc
        matching_geo = customers_sellers.merge(sellers_geo,
                                            on='seller_id')\
            .merge(customers_geo,
                   on='customer_id',
                   suffixes=('_seller',
                             '_customer'))
        # Remove na()
        matching_geo = matching_geo.dropna()

        matching_geo.loc[:, 'distance_seller_customer'] =\
            matching_geo.apply(lambda row:
                               haversine_distance(row['geolocation_lng_seller'],
                                                  row['geolocation_lat_seller'],
                                                  row['geolocation_lng_customer'],
                                                  row['geolocation_lat_customer']),
                               axis=1)
        # Since an order can have multiple sellers,
        # return the average of the distance per order
        order_distance =\
            matching_geo.groupby('order_id',
                                 as_index=False).agg({'distance_seller_customer':
                                                      'mean'})

        return order_distance

    def get_training_data(self,
                          is_delivered=True,
                          with_distance_seller_customer=False):
        """
        Returns a clean DataFrame (without NaN), with the all following columns:
        ['order_id', 'wait_time', 'expected_wait_time', 'delay_vs_expected',
        'order_status', 'dim_is_five_star', 'dim_is_one_star', 'review_score',
        'number_of_products', 'number_of_sellers', 'price', 'freight_value',
        'distance_seller_customer']
        """
        # Hint: make sure to re-use your instance methods defined above
        orders = self.get_wait_time(is_delivered=is_delivered)
        reviews = self.get_review_score()
        items = self.get_number_products()
        sellers = self.get_number_sellers()
        pricefreight = self.get_price_and_freight()
        df = (pd.merge(orders, reviews, on='order_id')
              .merge(items, on='order_id')
              .merge(sellers, on='order_id')
              .merge(pricefreight, on='order_id')
        )
        if with_distance_seller_customer == True:
            distance = self.get_distance_seller_customer()
            df = pd.merge(df, distance, on='order_id')
        return df.dropna()
