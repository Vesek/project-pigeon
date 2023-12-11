#project pigeon

print("hello")
answer = input("what is your name?: ")
pepa = True
if answer == "Pepa" or "Matyáš":
    print("hello pepa or matyas")
else:
    print("wrong name")
    

class human:
    def __init__(self, age, name) -> None:
        self.age = age
        self.name = name
        print(age)
        print(name)
    def eating(food):
        print("this human is eating ", food)
    
human(99, "Mikuláš")    

    





