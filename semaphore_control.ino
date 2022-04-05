//define output pins for relays 
#define SEM1pin 9 
#define SEM2pin 10
#define BTNpin 8


//globals
const int clDelay = 100*4;
bool sem1 = false;
bool sem2 = false;
int state = 0;

unsigned long tl, tc;
unsigned long changeTime = 1000*12;


void startup(){
  //semaphore 1 to red
  Serial.println("Starting up");
  if(sem1 == false && sem2 == false){
    change_state(SEM1pin, 1);
    sem1 = false;
    change_state(SEM2pin, 3);
    sem2 = true;
  }
  else{
    turnoff();
    startup();
  }
}

void turnoff(){
  //turn off sem 1
  Serial.println("Turn OFF!");
  if(sem1){
    change_state(SEM1pin, 1);
  }
  else{
    change_state(SEM1pin, 3);
  }
  //turn off sem2
  if(sem2){
    change_state(SEM2pin, 1);
  }
  else{
    change_state(SEM2pin, 3);
  }
  sem1 = false;
  sem2 = false;
}

void change_state(int pin, int n){
  for(int i=0;i<n;i++){
      delay(clDelay);
      
      digitalWrite(pin, HIGH);
      delay(clDelay);
      digitalWrite(pin, LOW);
  }
}

void change_sem(){
  if(sem1){
    change_state(SEM1pin, 2);
  }
  //turn off sem2
  if(sem2){
    change_state(SEM2pin, 2);
  }
  if(!sem1){
    change_state(SEM1pin, 2);
  }
  if(!sem2){
    change_state(SEM2pin, 2);
  }
  //change bools
  sem1 = !sem1;
  sem2 = !sem2;
}

void setup() {
  //pins init
  Serial.begin(9600);
  pinMode(SEM1pin, OUTPUT);
  pinMode(SEM2pin, OUTPUT);
  pinMode(BTNpin, INPUT_PULLUP);  
}

void loop() {  
  if(digitalRead(BTNpin) == LOW){
    Serial.println("Button Pressed!");
    if(state == 0){
      //if semaphores are off
      startup();
      state = 1;
    }
    else if(state == 1){
      turnoff();
      state = 0;
    }
    delay(1000);
  }
  else if(state == 1){
    tc = millis();
    if((tc-tl) > changeTime || tl>tc){
      Serial.println("Change sem!");
      change_sem();
      tl = millis();
    }
  }
}
