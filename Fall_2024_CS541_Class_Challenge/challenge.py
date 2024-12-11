### My implementationfor the challenge

from typing import Tuple
import numpy as np
import pandas as pd
import torch
from sklearn import tree, ensemble
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

'''
Notes about problem:
- regression problem
- train on input data, need to predict based on features
- evaluated on MSE of LOG prices

things to try:
- regression problem
  - linear regression
  - logistic regression
- neural network approaches -> pytorch
  - linear models + activation functions
  - residual connections?
- regression trees? -> sklearn
- boosting?
-

what doesn't make sense:
- recurrent neural networks (not sequential data, so it doesn't make a ton of sense)
- convolutional NNs (the data comes from different features, different features can't necessarily be connected together)

approaches tried:
- sklearn.tree.DecisionTreeRegressor
  - train log MSE: 0.0417093276433341
  - validation log MSE: 0.3586
  - notes: overfitting to training data!!
      this is likely a naive approach to solving this problem (overfitting), let's try to do cross validation
    on the training data to gain a better understanding of how each model is performing and how much it is
    overfitting

- sklearn.ensemble.GradientBoostingRegressor
  - train log MSE: 0.03463397524760063
  - validation log MSE: 0.1674
  - cross validation scores: [-0.13925216 -0.14871137 -0.13613333 -0.1396184  -0.14120412]
  - notes: trying to use gradient boosting to get things done

- linear feedforward neural networks with ReLU activation functions
  - train log MSE: 0.004442596359365533
  - validation log MSE: 0.263 (target is 0.1429)
  - notes: likely overfitting to training data due to such a low training log MSE (we'll see based on validation MSE)
    - can try to reduce overfitting by adding dropout and adding regularization?
  - with L2 regluarization
    - train log MSE: 0.007972340078077735

- feature engineering:
  - don't know how useful 

'''

class Model:
    def __init__(self):
        self.columns_of_interest = ['latitude', 'longitude', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'security_deposit', 'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'instant_bookable', 'is_business_travel_ready', 'Firm_mattress', 'Step-free_access', 'Dishes_and_silverware', '24-hour_check-in', 'Refrigerator', 'First_aid_kit', 'Pocket_wifi', 'Long_term_stays_allowed', 'Mobile_hoist', 'Self_check-in', 'Children’s_dinnerware', 'Beach_essentials', 'Gym', 'Air_purifier', 'Single_level_home', 'Hair_dryer', 'BBQ_grill', 'Changing_table', 'Building_staff', 'Paid_parking_off_premises', 'Hot_water_kettle', 'Suitable_for_events', 'Laptop_friendly_workspace', 'Roll-in_shower', 'Ground_floor_access', 'Cable_TV', 'Ski-in/Ski-out', 'Cooking_basics', 'Buzzer/wireless_intercom', 'Luggage_dropoff_allowed', 'Wide_clearance_to_bed', 'Garden_or_backyard', 'Fixed_grab_bars_for_toilet', 'Dog(s)', 'Keypad', 'Internet', 'Bathtub', 'Baby_bath', 'Flat_path_to_front_door', 'Shower_chair', 'Breakfast', 'Window_guards', 'Family/kid_friendly', 'Ceiling_hoist', 'Room-darkening_shades', 'Pool', 'Waterfront', 'Ethernet_connection', 'Fireplace_guards', 'Oven', 'Indoor_fireplace', 'Coffee_maker', 'Smoke_detector', 'Bed_linens', 'Wide_hallway_clearance', 'Host_greets_you', 'Private_living_room', 'Private_bathroom', 'Wide_entryway', 'Lake_access', 'Other_pet(s)', 'High_chair', 'Carbon_monoxide_detector', 'Safety_card', 'Hangers', 'Essentials', 'Iron', 'Smart_lock', 'Hot_water', 'Heating', 'Pool_with_pool_hoist', 'Pack_’n_Play/travel_crib', 'Well-lit_path_to_entrance', 'Accessible-height_bed', 'EV_charger', '_toilet', 'Dishwasher', 'Children’s_books_and_toys', 'Wide_doorway', 'Beachfront', 'Pets_live_on_this_property', 'Shampoo', 'Table_corner_guards', 'Fixed_grab_bars_for_shower', 'Washer', 'Outlet_covers', 'Bathtub_with_bath_chair', 'Free_parking_on_premises', 'Wifi', 'Lock_on_bedroom_door', 'Electric_profiling_bed', 'Stove', 'Lockbox', 'Wheelchair_accessible', 'Cleaning_before_checkout', 'Handheld_shower_head', 'Cat(s)', 'Fire_extinguisher', 'Air_conditioning', 'TV', 'Smoking_allowed', 'Baby_monitor', 'Extra_pillows_and_blankets', 'Game_console', 'Dryer', 'Disabled_parking_spot', 'Microwave', 'Stair_gates', 'Washer_/_Dryer', 'Patio_or_balcony', 'Paid_parking_on_premises', 'Pets_allowed', 'Babysitter_recommendations', 'Elevator', 'Accessible-height_toilet', 'Free_street_parking', 'Doorman', 'Wide_clearance_to_shower', 'Private_entrance', 'Hot_tub', 'Kitchen', 'Crib', 'Aparthotel', 'Apartment', 'Bed and breakfast', 'Boat', 'Boutique hotel', 'Bungalow', 'Cabin', 'Camper/RV', 'Casa particular (Cuba)', 'Castle', 'Cave', 'Condominium', 'Cottage', 'Guest suite', 'Guesthouse', 'Hostel', 'Hotel', 'House', 'Houseboat', 'Loft', 'Nature lodge', 'Resort', 'Serviced apartment', 'Tent', 'Timeshare', 'Tiny house', 'Townhouse', 'Train', 'Villa', 'Airbed', 'Couch', 'Futon', 'Pull-out Sofa', 'Real Bed', 'Entire home/apt', 'Private room', 'Shared room', 'comments']

        # try again with GradientBoostingRegressor with feature engineering
        self.model = ensemble.GradientBoostingRegressor(learning_rate=0.15, n_estimators=150)
        self.scalar = MinMaxScaler()

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        """
        Train model with training data.
        Currently, we use a linear regression with random weights
        You need to modify this function.
        :param X_train: shape (N,d)
        :param y_train: shape (N,1)
            where N is the number of observations, d is feature dimension
        :return: None
        """
        N, d = X_train.shape

        # train model
        new_data = self.scalar.fit_transform(X_train)

        self.model.fit(new_data[:, 6:], y_train)
        return None

    def predict(self, X_test: pd.DataFrame) -> np.array:
        """
        Use the trained model to predict on un-seen dataset
        You need to modify this function
        :param X_test: shape (N, d), where N is the number of observations, d is feature dimension
        return: prediction, shape (N,1)
        """
        data = self.scalar.transform(X_test)
        return self.model.predict(data[:, 6:])
