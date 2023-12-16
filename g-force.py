if __name__ == "__main__":
    import math
    from sense_hat import SenseHat

    

    sense = SenseHat()

    

   

    acceleration = sense.get_accelerometer_raw()

    x = acceleration['x']

    y = acceleration['y']

    z = acceleration['z']

    forcex = x
    forcey = y
    forcez = z
    m = 450000

    a = ((forcey / (math.sin(forcex / forcey) * math.sin(forcez * math.sin(forcex / forcey) / forcey))) / m)*1000000
    print("the overall acceleration of ISS is ",a,"(mikro metrů/s2)")
   

    
