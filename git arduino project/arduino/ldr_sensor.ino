/*
 * LDR Sensor Data Logger
 * ----------------------
 * Reads light intensity from an LDR connected to analog pin A0
 * and transmits readings over serial at 9600 baud.
 *
 * Circuit:
 *   - LDR one leg -> 5V
 *   - LDR other leg -> A0 and one leg of 10kΩ resistor
 *   - Other leg of 10kΩ resistor -> GND
 *
 * Output format (CSV): millis,ldr_value
 * Example: 1023,745
 */

const int LDR_PIN     = A0;   // Analog pin connected to LDR voltage divider
const int BAUD_RATE   = 9600;
const int SAMPLE_DELAY = 500; // Milliseconds between readings

void setup() {
  Serial.begin(BAUD_RATE);
  // Brief startup delay to allow serial monitor to connect
  delay(1000);
  // Send CSV header so the Python logger can parse cleanly
  Serial.println("timestamp_ms,ldr_value");
}

void loop() {
  int ldrValue = analogRead(LDR_PIN); // Range: 0–1023
  unsigned long timestamp = millis(); // Time since board powered on (ms)

  // Transmit as "timestamp_ms,ldr_value\n"
  Serial.print(timestamp);
  Serial.print(",");
  Serial.println(ldrValue);

  delay(SAMPLE_DELAY);
}
