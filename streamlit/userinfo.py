import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn


class Person:
    def __init__(self, gender, age, hypertension,
                 heart_disease, ever_married, work_type, Residence_type,
                 avg_glucose_level, bmi, smoking_status):
    self.gender = gender
    self.age = age
    self.hypertension = hypertension
    self.heart_disease = heart_disease
    self.ever_married = ever_married
    self.work_type = work_type
    self.Residence_type = Residence_type
    self.avg_glucose_level = avg_glucose_level
    self.bmi = bmi
    self.smoking_status = smoking_status


def myfunc(self):
    print("Hello my name is " + self.gender




if __name__ == '__main__':
    p1=Person()
