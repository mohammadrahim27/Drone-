% component weights from Excel (units = grams)
function [auw, htpm, mhc, tc, tcc, bc, ttw, esc_result, xt60h_result, pm02_pt_result, pm02_5v_result, battery_result, ttw_result] = variables()
    % individual component weights in grams
    fc = 62.4;
    esc = 13.8; % 4-in-1 esc
    motor = 47.59; % x4, 1300KV
    battery = 396;
    propellar = 8.8; % x4
    reciever = 4;
    gps = 32;
    frame = 380;
    companion_controller = 174; % Jetson Orin Nano Super Developer Kit
    misc = 60; % battery lead, XT60 connectors, capacitors, screws, straps, etc.
    
    % all up weight (AUW)
    auw = fc + esc + (motor * 4) + battery + (propellar * 4) + reciever + gps + frame + companion_controller + misc;

    % motor hover current (24V), 30% thrust cap (1300KV)
    thrust_lower = 310;
    thrust_upper = 530;
    current_lower = 2.3;
    current_upper = 4.6;
    motor_count = 4;
    htpm = auw/motor_count; % hover thrust per motor

    if htpm < 310
        hover_current = 2.3;
    elseif htpm > 310
        hover_current = current_lower + (htpm - thrust_lower) * (current_upper - current_lower) / (thrust_upper - thrust_lower);
    end

    mhc = hover_current * 4;

    % current limits at 5V (conservative)
    fc_mcd = 0.42; % FC max current draw
    gps_mcd = 0.2; % GPS max current draw
    r_mcd = 0.15; % Receiver max current draw (PLACEHOLDER)

    ec_5v = fc_mcd + gps_mcd + r_mcd; % max current drawn by electronics at 5V (not including mhc)
    power_5v = 5 * ec_5v;
    bv = 22.2; % battery voltage
    bc = power_5v / bv; % battery current -- re-scaled current limit at 5V to 22.2V
    mcc = 2.2 * 70; % max continuous current = battery capacity (Ah) * C rating -- 2.2 Ah, 70C

    % 50% Thrust cap (1300KV)
    tc = 750 * 4; % thrust cap, units = gf
    tcc = 7.2 * 4; % thrust cap current units = amp -- aim for no more than 80% of 30A
    
    % Thrust to Weight Ratio
    ttw = tc/auw; % at least 2:1 ratio

    % ESC Check
    if tcc/4 < 50 
        esc_result = "ESC Pass ";
    else
        esc_result = "ESC FAIL ";
    end

    % XT60H Check
    if tcc < 30
        xt60h_result = "XT60H Pass ";
    else
        xt60h_result = "XT60H FAIL ";
    end
    
    % PM02 Pass-Through Current Check
    if tcc < 60
        pm02_pt_result = "PM02 PT Pass ";
    else
        pm02_pt_result = "PM02 PT FAIL ";
    end
    
    % PM02 5V Output Current Check
    if ec_5v < 3
        pm02_5v_result = "PM02 5V Pass ";
    else
        pm02_5v_result = "PM02 5V FAIL ";
    end
    
    % Battery Check
    if tcc < mcc
        battery_result = "Battery Pass ";
    else
        battery_result = "Battery FAIL ";
    end

    % Thrust to Weight Ratio Check
    if ttw > 2
        ttw_result = "TTW Pass ";
    else
        ttw_result = "TTW FAIL ";
    end
end