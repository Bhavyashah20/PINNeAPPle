model DrainingTank "Tank with a constant inflow, emptying through an orifice (Torricelli)"
  parameter Real A = 2.0 "tank cross-section [m2]";
  parameter Real a = 0.01 "orifice area [m2]";
  parameter Real Cd = 0.62 "discharge coefficient [-]";
  parameter Real qin = 0.01 "inflow [m3/s]";
  parameter Real h0 = 1.0 "initial level [m]";
  constant Real g = 9.81;
  Real h(start = h0, fixed = true) "level [m]";
  output Real level = h;
  Real qout "outflow [m3/s]";
equation
  qout = Cd * a * sqrt(2 * g * max(h, 0));
  A * der(h) = qin - qout;
end DrainingTank;
