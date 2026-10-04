# S E N S O R S  
1. Ultrasonic (waterproof)
2. Rain
3. Temperature
4. Humidity
5. Soil Moisture
6. Water Flow

# D A T A    F O R M A T
1. Time Stamp Sensor Readings = "2026-07-29 13:12:00" (SQL style)
2. Ultrasonic Readings = meters
3. Rain Sensor = millimeters (1 mm = 1 L/m^2)
4. Temperture = Degree Celius (°C)
5. Humidity = Percentage (%)
6. Soil Moisture = Percentage (%)
7. Water Flow = Liters per minute (L/min)

# D A T A S E T
Contains 7 columns
Contains 10 11 rows in total

Informal Size Categories
- Small Data: 100 to a few thousands
- Medium Data: tens of thousands to hundred of thousands of rows
- Large Data: millions or billions of rows


To accurately predict river water level height, you generally need a minimum of 2 to 5 years of continuous historical data (approximatelt 17,500 to 43,000 hourly rows)

Traditional Machine Learning Model needs 10,000 to 30,000 rows

# LABELED DATA VS UNLABELED DATA
- Labeled means the sensor readings plus actual reading from the site
- Unlabeled refers solely to the sensor readings accumulated without comparing the actual water on site