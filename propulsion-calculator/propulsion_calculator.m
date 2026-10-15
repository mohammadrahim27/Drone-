[auw, htpm, mhc, tc, tcc, bc, ttw, esc_result, xt60h_result, pm02_pt_result, pm02_5v_result, battery_result, ttw_result] = variables();
motor_count = 4; % number of motors used by drone 
total_current = mhc + bc;
usable_capacity = 2.2 * 0.8; % don't use more than 80% of the battery's 2.2 amp hours(Ah)
flight_time = (usable_capacity / total_current) * 60; % units = min

% Safety Checks
disp(esc_result + xt60h_result + pm02_pt_result)
disp(pm02_5v_result + battery_result + ttw_result)

% Summary
disp("AUW: " + auw + "g | Thrust Cap: 50% | Usable Battery Capacity: 80% of 2.2Ah")
disp("Thrust per motor: " + htpm + "gf")
disp("Thrust to weight ratio: " + ttw)
disp("Motor hover current: " + mhc + "A")
disp("Electronics current (battery-side): " + bc + "A")
disp("Total hover current: " + total_current + "A")
disp("Current drawn at thrust cap: " + tcc + "A")
disp("Flight time: " + flight_time + "min")



torch.save