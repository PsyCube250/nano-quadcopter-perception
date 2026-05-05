//OLED display code for arduino

#include <Arduino.h>
#include <U8g2lib.h>
#include <SPI.h>

/*
  DISPLAY PINOUT (Standard 1-7 sequential):
  1: GND     -> GND
  2: VCC     -> 3.3V
  3: D0/SCK  -> Pin 13 (SPI Clock)
  4: D1/SDA  -> Pin 11 (SPI MOSI)
  5: RES/RST -> Pin 9  (Reset)
  6: DC      -> Pin 8  (Data/Command)
  7: CS      -> Pin 10 (Chip Select)
*/

// Constructor for SSD1309 4-wire Hardware SPI
U8G2_SSD1309_128X64_NONAME0_F_4W_HW_SPI u8g2(U8G2_R0, /* cs=*/ 10, /* dc=*/ 8, /* reset=*/ 9);

void setup() {
  // Start the display
  u8g2.begin();
}

void loop() {
  char PiCamStatus[5] = "Good";
  char AutoPilotStatus[4] = "Off";
  int BatteryCharge = 100;

  u8g2.clearBuffer();					// Clear internal memory
  u8g2.setFont(u8g2_font_6x10_tf);	// Choose a font

  u8g2.drawStr(0, 20, " PiCam Status: ");	// Write text
  u8g2.setCursor(90, 20);   // 3. Use .print() to display the variable
  u8g2.print(PiCamStatus);

  u8g2.drawStr(0, 35, " Auto Pilot Status: ");	// Write text
  u8g2.setCursor(120, 35);   // 3. Use .print() to display the variable
  u8g2.print(AutoPilotStatus);

  u8g2.drawStr(0, 50, " Battery Charge: ");	// Write text
  u8g2.setCursor(102, 50);   // 3. Use .print() to display the variable
  u8g2.print(BatteryCharge);
  u8g2.print("%");

  u8g2.drawFrame(0, 0, 128, 64);        // Draw a border
  
  u8g2.sendBuffer();					// Send memory to the display
  
  delay(1000);
}
