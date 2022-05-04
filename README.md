# Flight delay prediction

Predict airline delays for Tunisian aviation company, Tunisair

---

## The data

Flight Data:
There is data from over 100.000 flights from Tunis Air from the years 2016 - 2018.
https://zindi.africa/competitions/ai-tunisia-hack-5-predictive-analytics-challenge-2

Airport Data: 
There is data on airports worldwide, their location and coordinates.
https://pypi.org/project/airportsdata/

---

## Description

Flight delays not only irritate air passengers and disrupt their schedules but also cause :

* a decrease in efficiency
* an increase in capital costs, reallocation of flight crews and aircraft
* an additional crew expenses

As a result, on an aggregate basis, an airline's record of flight delays may have a negative impact on passenger demand.
This solution proposes to build a flight delay predictive model using Machine Learning techniques. The accurate prediction of flight delays will help all players in the air travel ecosystem to set up effective action plans to reduce the impact of the delays and avoid loss of time, capital and resources.

---

## Business Understanding

About the stakeholder: Tunisair

Tunisair is the flag carrier airline of Tunisia. Formed in 1948, it operates scheduled international services to four continents. Its main base is Tunis–Carthage International Airport. The airline's head office is in Tunis, near Tunis Airport. Tunisair is a member of the Arab Air Carriers Organization

---

## Setting up Problem

Lets set some background first:

Customer browsing through some flight booking website and want to book flight for some specific date, time, source and destination.

__IDEA:__ If during flight booking, we can show to customer whether the flight he/she considering for booking is likely to arrive on time or not. Additionaly, if flight is expected to delay, also show delayed time.

If customer know that the flight is likely to be late, he/she might choose to book another flight.

From Modelling Propective, need to set two goals:

__GOAL I:__ Predict whether flight is going to delay or not.
<br>

__GOAL II:__ If flight delays, predict amount of time by which it delays.


---

## Requirements and Environment

Requirements:
- pyenv with Python: 3.9.8

Environment: 

For installing the virtual environment you can either use the Makefile and run `make setup` or install it manually with the following commands: 

```Bash
pyenv local 3.9.8
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
