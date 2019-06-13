#include <DHT.h>
#include <DHT_U.h>
DHT Q(13,DHT11);
void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  Q.begin();
  Serial.print("Time and date,Temp in F,Temp in C,Humidity");
}

void loop() {
  float f=Q.readTemperature(true);
  delay(2000);
  float h=Q.readTemperature();
  delay(2000000);
  float g=Q.readHumidity(true);
  delay(2000000);
  Serial.print(",");
  Serial.print(f);
  Serial.print(",");
  Serial.print(h);
  Serial.print(",");
  Serial.println(g);
  
}
