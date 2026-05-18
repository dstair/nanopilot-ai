var driveHandler = new function() {
    //functions used to drive the vehicle. 

    var state = {
        'tele': {
            "user": {
                'angle': 0,
                'throttle': 0,
                'roll': 0,
                'altitude': 0,
            },
            "pilot": {
                'angle': 0,
                'throttle': 0,
                'roll': 0,
                'altitude': 0,
            }
        },
        'rc': {
            'roll': 1500,
            'pitch': 1500,
            'yaw': 1500,
            'throttle': 1000,
            'arm': 1000,
            'mode': 2000,
        },
        'bf': {
            'armed': false,
            'arming_flags': null,
            'arming_disable_flags': '',
            'active_modes': '',
        },
        'config': {
            'max_pitch_angle': null,
            'max_roll_angle': null,
            'hover_throttle': null,
            'throttle_range': null,
            'input_sensitivity': null,
            'max_yaw_rate': null,
            'yaw_pwm_cap': null,
            'yaw_throttle_feedforward': null,
            'pitch_throttle_feedforward': null,
            'altitude_hold_k': null,
            'altitude_hold_deadband': null,
        },
        'brakeOn': true,
        'recording': false,
        'driveMode': "user",
        'pilot': 'None',
        'session': 'None',
        'lag': 0,
        'controlMode': 'joystick',
        'maxThrottle' : 1,
        'throttleMode' : 'user',
        'buttons': {
            "w1": false,  // boolean; true is 'down' or pushed, false is 'up' or not pushed
            "w2": false,
            "w3": false,
            "w4": false,
            "w5": false,
        }
    }

    var joystick_options = {}
    var joystickLoopRunning=false;

    var hasGamepad = false;

    var deviceHasOrientation=false;
    var initialGamma;

    var vehicle_id = ""
    var driveURL = ""
    var socket

    this.load = function() {
      driveURL = '/drive'
      socket = new WebSocket('ws://' + location.host + '/wsDrive');

      applyDroneTheme()
      injectRcPanel()
      setBindings()

      joystick_element = document.getElementById('joystick_container');
      joystick_options = {
        zone: joystick_element,  // active zone
        mode: 'dynamic',
        size: 200,
        color: '#668AED',
        dynamicPage: true,
        follow: true,
      };

      var manager = nipplejs.create(joystick_options);
      bindNipple(manager)

      if(!!navigator.getGamepads){
        console.log("Device has gamepad support.")
        hasGamepad = true;
      }

      if (window.DeviceOrientationEvent) {
        window.addEventListener("deviceorientation", handleOrientation);
        console.log("Browser supports device orientation, setting control mode to tilt.");
        state.controlMode = 'tilt';
        deviceOrientationLoop();
      } else {
        console.log("Device Orientation not supported by browser, setting control mode to joystick.");
        state.controlMode = 'joystick';
      }
    };

    //
    // Update a state object with the given data.
    // This will only update existing fields in 
    // the state; it will not add new fields that
    // may exist in the data but not the state.
    //
    var updateState = function(state, data) {
        let changed = false;
        if(typeof data === 'object') {
            const keys = Object.keys(data)
            keys.forEach(key => {
                //
                // state must already have the key;
                // we are not adding new fields to the state,
                // we are only updating existing fields.
                //
                if(state.hasOwnProperty(key) && state[key] !== data[key]) {
                    if(typeof state[key] === 'object') {
                        // recursively update the state's object field
                        if(updateState(state[key], data[key])) changed = true;
                    } else {
                        state[key] = data[key];
                        changed = true;
                    }
                }
            });
        }
        return changed;
    }

    var setBindings = function() {
      //
      // when server sends a message with state changes
      // then update our local state and 
      // if there were any changes then redraw the UI.
      //
      socket.onmessage = function (event) {
        console.log(event.data);
        const data = JSON.parse(event.data);
        if(updateState(state, data)) {
            updateUI();
        }
      };

      $(document).keydown(function(e) {
          if(e.which == 32) { toggleBrake() }  // 'space'  brake
          if(e.which == 82) { toggleRecording() }  // 'r'  toggle recording
          if(e.which == 73) { throttleUp() }  // 'i'  throttle up
          if(e.which == 75) { throttleDown() } // 'k'  slow down
          if(e.which == 37) { angleLeft(); e.preventDefault() } // left arrow - yaw left
          if(e.which == 39) { angleRight(); e.preventDefault() } // right arrow - yaw right
          if(e.which == 74) { rollLeft() } // 'j' roll left
          if(e.which == 76) { rollRight() } // 'l' roll right
          if(e.which == 38) { altitudeUp(); e.preventDefault() } // arrow up - altitude up
          if(e.which == 40) { altitudeDown(); e.preventDefault() } // arrow down - altitude down
          if(e.which == 65) { updateDriveMode('local') } // 'a' turn on local mode (full _A_uto)
          if(e.which == 85) { updateDriveMode('user') } // 'u' turn on manual mode (_U_user)
          if(e.which == 83) { updateDriveMode('local_angle') } // 's' turn on local mode (auto _S_teering)
          if(e.which == 77) { toggleDriveMode() } // 'm' toggle drive mode (_M_ode)
      });

      // Release flight-control keys → axes recenter to neutral. Analog-stick feel.
      $(document).keyup(function(e) {
          if(e.which == 73 || e.which == 75) { throttleCenter(); e.preventDefault() }
          if(e.which == 38 || e.which == 40) { altitudeCenter(); e.preventDefault() }
          if(e.which == 37 || e.which == 39) { angleCenter(); e.preventDefault() }
          if(e.which == 74 || e.which == 76) { rollCenter(); e.preventDefault() }
      });

      $('#mode_select').on('change', function () {
        updateDriveMode($(this).val());
      });

      $('#max_throttle_select').on('change', function () {
        state.maxThrottle = parseFloat($(this).val());
      });

      $('#throttle_mode_select').on('change', function () {
        state.throttleMode = $(this).val();
      });

      $('#record_button').click(function () {
        toggleRecording();
      });

      $('#brake_button').click(function() {
        toggleBrake();
      });

      $('input[type=radio][name=controlMode]').change(function() {
        if (this.value == 'joystick') {
          state.controlMode = "joystick";
          joystickLoopRunning = true;
          console.log('joystick mode');
          joystickLoop();
        } else {
          joystickLoopRunning = false;
        }

        if (deviceHasOrientation && this.value == 'tilt') {
          state.controlMode = "tilt";
          console.log('tilt mode')
        }

        if (hasGamepad && this.value == 'gamepad') {
          state.controlMode = "gamepad";
          console.log('gamepad mode')
          gamePadLoop();
        }
        updateUI();
      });

      // programmable buttons
      $('#button_bar > button').mousedown(function() {
        console.log(`${$(this).attr('id')} mousedown`);
        state.buttons[$(this).attr('id')] = true;
        postDrive(["buttons"]); // write it back to the server
      });
      $('#button_bar > button').mouseup(function() {
        console.log(`${$(this).attr('id')} mouseup`);
        state.buttons[$(this).attr('id')] = false;
        postDrive(["buttons"]); // write it back to the server
      });
    };


    function bindNipple(manager) {
      manager.on('start', function(evt, data) {
        state.tele.user.angle = 0
        state.tele.user.throttle = 0
        state.recording = true
        joystickLoopRunning=true;
        joystickLoop();

      }).on('end', function(evt, data) {
        joystickLoopRunning=false;
        brake()

      }).on('move', function(evt, data) {
        state.brakeOn = false;
        radian = data['angle']['radian']
        distance = data['distance']

        //console.log(data)
        state.tele.user.angle = Math.max(Math.min(Math.cos(radian)/70*distance, 1), -1)
        state.tele.user.throttle = limitedThrottle(Math.max(Math.min(Math.sin(radian)/70*distance , 1), -1))

        // DonkeyCar car logic: zero steering when stopped. Disabled for drone
        // (drone can yaw in place).
        // if (state.tele.user.throttle < .001) {
        //   state.tele.user.angle = 0
        // }

      });
    }

    var applyDroneTheme = function() {
      if ($('#nanopilot-theme').length) return;
      var css = ''
        + 'html, body { background:#000 !important; color:#eee !important; }'
        + '.container, .container-fluid, .row, #content, #main, #drive, #control-bars {'
        +   'background:#000 !important;'
        + '}'
        + 'label, .control-label, .form-check-label { color:#eee !important; }'
        + 'a { color:#8ab4ff; }'
        + '#control-bars { display:none !important; }'
        + '#rc-panel { margin-top:10px;padding:14px;background:#202020;color:#eee;'
        +   'border-radius:4px;font-family:Menlo,Consolas,monospace;'
        +   'box-shadow:inset 0 0 0 1px rgba(255,255,255,.08); }'
        + '#rc-panel * { box-sizing:border-box; }'
        + '.rc-header { display:flex;align-items:center;justify-content:space-between;gap:12px;'
        +   'padding-bottom:10px;border-bottom:1px solid rgba(255,255,255,.10); }'
        + '.rc-source { color:#f0f0f0;font-size:18px;font-weight:700; }'
        + '.rc-status { min-width:118px;text-align:center;padding:5px 10px;border-radius:4px;'
        +   'font-size:18px;font-weight:800;letter-spacing:.04em; }'
        + '.rc-status.armed { background:#471b1b;color:#ff8a8a;box-shadow:inset 0 0 0 1px #cf5555; }'
        + '.rc-status.disarmed { background:#18281f;color:#78e5a0;box-shadow:inset 0 0 0 1px #3c8f58; }'
        + '.rc-meta { display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:10px 0 12px;'
        +   'font-size:13px;color:#ccc; }'
        + '.rc-meta span { color:#fff; }'
        + '.rc-sticks { display:grid;grid-template-columns:1fr 1fr;gap:12px; }'
        + '.rc-stick-card { min-width:0; }'
        + '.rc-stick-title { display:flex;justify-content:space-between;gap:8px;margin-bottom:6px;'
        +   'font-size:13px;color:#ddd; }'
        + '.rc-stick-title strong { color:#fff; }'
        + '.rc-stick { position:relative;width:100%;aspect-ratio:1 / 1;min-height:142px;'
        +   'border-radius:4px;background:#101010;overflow:hidden;'
        +   'box-shadow:inset 0 0 0 1px rgba(255,255,255,.16); }'
        + '.rc-stick:before, .rc-stick:after { content:"";position:absolute;background:rgba(255,255,255,.16); }'
        + '.rc-stick:before { left:50%;top:8px;bottom:8px;width:1px; }'
        + '.rc-stick:after { top:50%;left:8px;right:8px;height:1px; }'
        + '.rc-stick-fill { position:absolute;left:0;right:0;bottom:0;height:0%;'
        +   'background:linear-gradient(0deg,rgba(108,204,255,.30),rgba(108,204,255,.06)); }'
        + '.rc-stick-dot { position:absolute;left:50%;top:50%;width:18px;height:18px;margin:-9px 0 0 -9px;'
        +   'border-radius:50%;background:#f5f5f5;box-shadow:0 0 0 3px rgba(102,138,237,.35);'
        +   'transition:left .08s linear,top .08s linear; }'
        + '.rc-axis-label { position:absolute;color:#aaa;font-size:11px;line-height:1;pointer-events:none; }'
        + '.rc-axis-label.top { top:8px;left:50%;transform:translateX(-50%); }'
        + '.rc-axis-label.bottom { bottom:8px;left:50%;transform:translateX(-50%); }'
        + '.rc-axis-label.left { left:8px;top:50%;transform:translateY(-50%); }'
        + '.rc-axis-label.right { right:8px;top:50%;transform:translateY(-50%); }'
        + '.rc-values { display:grid;grid-template-columns:1fr 1fr;gap:6px 10px;margin-top:10px;'
        +   'font-size:13px; }'
        + '.rc-value { display:flex;justify-content:space-between;gap:8px;color:#bbb; }'
        + '.rc-value span { color:#fff; }'
        + '.rc-config { margin-top:12px;padding-top:10px;border-top:1px solid rgba(255,255,255,.10); }'
        + '.rc-config-title { color:#fff;font-size:13px;font-weight:700;margin-bottom:6px; }'
        + '.rc-config-grid { display:grid;grid-template-columns:1fr 1fr;gap:6px 10px;font-size:12px; }'
        + '.rc-config-item { display:flex;justify-content:space-between;gap:8px;color:#aaa; }'
        + '.rc-config-item span { color:#f2f2f2; }'
        + '@media (max-width: 700px) {'
        +   '.rc-header { align-items:flex-start;flex-direction:column; }'
        +   '.rc-status { width:100%; }'
        +   '.rc-meta, .rc-sticks, .rc-values, .rc-config-grid { grid-template-columns:1fr; }'
        + '}';
      $('head').append('<style id="nanopilot-theme">' + css + '</style>');
    };

    var injectRcPanel = function() {
      if ($('#rc-panel').length) return;
      var html = ''
        + '<div id="rc-panel">'
        +   '<div class="rc-header">'
        +     '<div class="rc-source">BetaFlight</div>'
        +     '<div id="bf-status" class="rc-status disarmed">DISARMED</div>'
        +   '</div>'
        +   '<div class="rc-meta">'
        +     '<div>active modes: <span id="bf-active-modes">----</span></div>'
        +     '<div>arming flags: <span id="bf-arming-flags">----</span></div>'
        +   '</div>'
        +   '<div class="rc-sticks">'
        +     '<div class="rc-stick-card">'
        +       '<div class="rc-stick-title"><strong>Left stick</strong><span>throttle / yaw</span></div>'
        +       '<div class="rc-stick" id="rc-left-stick">'
        +         '<div id="rc-throttle-fill" class="rc-stick-fill"></div>'
        +         '<div class="rc-axis-label top">throttle</div>'
        +         '<div class="rc-axis-label bottom">idle</div>'
        +         '<div class="rc-axis-label left">yaw</div>'
        +         '<div class="rc-axis-label right">yaw</div>'
        +         '<div id="rc-left-dot" class="rc-stick-dot"></div>'
        +       '</div>'
        +     '</div>'
        +     '<div class="rc-stick-card">'
        +       '<div class="rc-stick-title"><strong>Right stick</strong><span>pitch / roll</span></div>'
        +       '<div class="rc-stick" id="rc-right-stick">'
        +         '<div class="rc-axis-label top">pitch</div>'
        +         '<div class="rc-axis-label bottom">back</div>'
        +         '<div class="rc-axis-label left">roll</div>'
        +         '<div class="rc-axis-label right">roll</div>'
        +         '<div id="rc-right-dot" class="rc-stick-dot"></div>'
        +       '</div>'
        +     '</div>'
        +   '</div>'
        +   '<div class="rc-values">'
        +     '<div class="rc-value">roll <span id="rc-roll">----</span></div>'
        +     '<div class="rc-value">pitch <span id="rc-pitch">----</span></div>'
        +     '<div class="rc-value">yaw <span id="rc-yaw">----</span></div>'
        +     '<div class="rc-value">throttle <span id="rc-throttle">----</span></div>'
        +     '<div class="rc-value">arm aux <span id="rc-arm">----</span></div>'
        +     '<div class="rc-value">mode aux <span id="rc-mode">----</span></div>'
        +   '</div>'
        +   '<div class="rc-config">'
        +     '<div class="rc-config-title">Flight config</div>'
        +     '<div class="rc-config-grid">'
        +       '<div class="rc-config-item">max pitch <span id="cfg-max-pitch">----</span></div>'
        +       '<div class="rc-config-item">max roll <span id="cfg-max-roll">----</span></div>'
        +       '<div class="rc-config-item">hover PWM <span id="cfg-hover-throttle">----</span></div>'
        +       '<div class="rc-config-item">throttle range <span id="cfg-throttle-range">----</span></div>'
        +       '<div class="rc-config-item">input sensitivity <span id="cfg-input-sensitivity">----</span></div>'
        +       '<div class="rc-config-item">max yaw rate <span id="cfg-max-yaw-rate">----</span></div>'
        +       '<div class="rc-config-item">yaw PWM cap <span id="cfg-yaw-pwm-cap">----</span></div>'
        +       '<div class="rc-config-item">yaw FF <span id="cfg-yaw-ff">----</span></div>'
        +       '<div class="rc-config-item">pitch FF <span id="cfg-pitch-ff">----</span></div>'
        +       '<div class="rc-config-item">alt hold K <span id="cfg-alt-hold-k">----</span></div>'
        +     '</div>'
        +   '</div>'
        + '</div>';
      var anchor = $('#control-bars');
      if (anchor.length) {
        anchor.after(html);
      } else {
        $('body').prepend(html);
      }
    };

    var clamp = function(value, min, max) {
      return Math.max(min, Math.min(max, value));
    };

    var normalizedPwm = function(value, center, span) {
      return clamp((parseFloat(value) - center) / span, -1, 1);
    };

    var updateStick = function(selector, x, y) {
      $(selector).css({
        left: ((x + 1) * 50) + '%',
        top: ((1 - (y + 1) / 2) * 100) + '%',
      });
    };

    var fmtConfig = function(value, suffix) {
      if (value === null || value === undefined || value === '') return '----';
      if (typeof value === 'number') {
        var rounded = Math.round(value * 100) / 100;
        return rounded + (suffix || '');
      }
      return value + (suffix || '');
    };

    var updateConfigUI = function() {
      $('#cfg-max-pitch').text(fmtConfig(state.config.max_pitch_angle, '°'));
      $('#cfg-max-roll').text(fmtConfig(state.config.max_roll_angle, '°'));
      $('#cfg-hover-throttle').text(fmtConfig(state.config.hover_throttle));
      $('#cfg-throttle-range').text('±' + fmtConfig(state.config.throttle_range));
      $('#cfg-input-sensitivity').text(fmtConfig(state.config.input_sensitivity));
      $('#cfg-max-yaw-rate').text(fmtConfig(state.config.max_yaw_rate, '°/s'));
      $('#cfg-yaw-pwm-cap').text(fmtConfig(state.config.yaw_pwm_cap));
      $('#cfg-yaw-ff').text(fmtConfig(state.config.yaw_throttle_feedforward, ' PWM'));
      $('#cfg-pitch-ff').text(fmtConfig(state.config.pitch_throttle_feedforward, ' PWM'));
      $('#cfg-alt-hold-k').text(fmtConfig(state.config.altitude_hold_k, ' PWM/(m/s)'));
    };

    var updateRcUI = function() {
      $('#rc-roll').text(state.rc.roll);
      $('#rc-pitch').text(state.rc.pitch);
      $('#rc-yaw').text(state.rc.yaw);
      $('#rc-throttle').text(state.rc.throttle);
      $('#rc-arm').text(state.rc.arm);
      $('#rc-mode').text(state.rc.mode);

      var status = state.bf.armed ? 'ARMED' : 'DISARMED';
      $('#bf-status')
        .text(status)
        .toggleClass('armed', state.bf.armed)
        .toggleClass('disarmed', !state.bf.armed);
      $('#bf-active-modes').text(state.bf.active_modes || '----');
      var flags = state.bf.arming_disable_flags || '';
      if (!flags && state.bf.arming_flags === 0) {
        flags = 'none';
      }
      $('#bf-arming-flags').text(flags || '----');

      var yaw = normalizedPwm(state.rc.yaw, 1500, 500);
      var roll = normalizedPwm(state.rc.roll, 1500, 500);
      var pitch = normalizedPwm(state.rc.pitch, 1500, 500);
      var throttle = clamp((parseFloat(state.rc.throttle) - 1000) / 1000, 0, 1);

      updateStick('#rc-left-dot', yaw, (throttle * 2) - 1);
      updateStick('#rc-right-dot', roll, pitch);
      $('#rc-throttle-fill').css('height', (throttle * 100) + '%');
      updateConfigUI();
    };

    var updateUI = function() {
      $("#throttleInput").val(state.tele.user.throttle);
      $("#angleInput").val(state.tele.user.angle);
      $('#mode_select').val(state.driveMode);
      updateRcUI();

      var throttlePercent = Math.round(Math.abs(state.tele.user.throttle) * 100) + '%';
      var steeringPercent = Math.round(Math.abs(state.tele.user.angle) * 100) + '%';
      var throttleRounded = state.tele.user.throttle.toFixed(2)
      var steeringRounded = state.tele.user.angle.toFixed(2)

      $('#throttle_label').html(throttleRounded);
      $('#steering_label').html(steeringRounded);

      if(state.tele.user.throttle < 0) {
        $('#throttle-bar-backward').css('width', throttlePercent).html(throttleRounded)
        $('#throttle-bar-forward').css('width', '0%').html('')
      }
      else if (state.tele.user.throttle > 0) {
        $('#throttle-bar-backward').css('width', '0%').html('')
        $('#throttle-bar-forward').css('width', throttlePercent).html(throttleRounded)
      }
      else {
        $('#throttle-bar-forward').css('width', '0%').html('')
        $('#throttle-bar-backward').css('width', '0%').html('')
      }

      if(state.tele.user.angle < 0) {
        $('#angle-bar-backward').css('width', steeringPercent).html(steeringRounded)
        $('#angle-bar-forward').css('width', '0%').html('')
      }
      else if (state.tele.user.angle > 0) {
        $('#angle-bar-backward').css('width', '0%').html('')
        $('#angle-bar-forward').css('width', steeringPercent).html(steeringRounded)
      }
      else {
        $('#angle-bar-forward').css('width', '0%').html('')
        $('#angle-bar-backward').css('width', '0%').html('')
      }

      if (state.recording) {
        $('#record_button')
          .html('Stop Recording (r)')
          .removeClass('btn-info')
          .addClass('btn-warning').end()
      } else {
        $('#record_button')
          .html('Start Recording (r)')
          .removeClass('btn-warning')
          .addClass('btn-info').end()
      }

      if (state.brakeOn) {
        $('#brake_button')
          .html('Start Vehicle')
          .removeClass('btn-danger')
          .addClass('btn-success').end()
      } else {
        $('#brake_button')
          .html('Stop Vehicle')
          .removeClass('btn-success')
          .addClass('btn-danger').end()
      }

      if(deviceHasOrientation) {
        $('#tilt-toggle').removeAttr("disabled")
        $('#tilt').removeAttr("disabled")
      } else {
        $('#tilt-toggle').attr("disabled", "disabled");
        $('#tilt').prop("disabled", true);
      }

      if(hasGamepad) {
        $('#gamepad-toggle').removeAttr("disabled")
        $('#gamepad').removeAttr("disabled")
      } else {
        $('#gamepad-toggle').attr("disabled", "disabled");
        $('#gamepad').prop("disabled", true);
      }

      if (state.controlMode == "joystick") {
        $('#joystick_outer').show();
        $('#joystick-toggle').addClass("active");
        $('#joystick').attr("checked", "checked")
      } else {
        $('#joystick_outer').hide();
        $('#joystick-toggle').removeClass("active");
        $('#joystick').removeAttr("checked");
      }

      if (state.controlMode == "tilt") {
        $('#tilt-toggle').addClass("active");
        $('#tilt').attr("checked", "checked");
      } else {
        $('#tilt-toggle').removeClass("active");
        $('#tilt').removeAttr("checked")
      }

      //drawLine(state.tele.user.angle, state.tele.user.throttle)
    };

    const ALL_POST_FIELDS = ['angle', 'throttle', 'roll', 'altitude', 'drive_mode', 'recording', 'buttons'];

    //
    // Set any changed properties to the server
    // via the websocket connection
    //
    var postDrive = function(fields=[]) {

        if(fields.length === 0) {
            fields = ALL_POST_FIELDS;
        }

        let data = {}
        fields.forEach(field => {
            switch (field) {
                case 'angle': data['angle'] = state.tele.user.angle; break;
                case 'throttle': data['throttle'] = state.tele.user.throttle; break;
                case 'roll': data['roll'] = state.tele.user.roll; break;
                case 'altitude': data['altitude'] = state.tele.user.altitude; break;
                case 'drive_mode': data['drive_mode'] = state.driveMode; break;
                case 'recording': data['recording'] = state.recording; break;
                case 'buttons': data['buttons'] = state.buttons; break;
                default: console.log(`Unexpected post field: '${field}'`); break;
            }
        });
        if(data) {
            let json_data = JSON.stringify(data);
            console.log(`Posting ${json_data}`);
            socket.send(json_data)
            updateUI()
        }
    };

    var applyDeadzone = function(number, threshold){
       percentage = (Math.abs(number) - threshold) / (1 - threshold);

       if(percentage < 0)
          percentage = 0;

       return percentage * (number > 0 ? 1 : -1);
    }



    // Mode 2 quadcopter stick layout (Xbox/PS controllers via HTML5 Gamepad API):
    //   left stick X  (axes[0]) → yaw      (steering)
    //   left stick Y  (axes[1]) → altitude (up = climb)
    //   right stick X (axes[2]) → roll
    //   right stick Y (axes[3]) → pitch    (forward = throttle)
    //   A button (buttons[0])   → brake (zero controls + brakeOn)
    //   Y button (buttons[3])   → cycle drive mode (rising edge)
    var prevYButton = false;
    function gamePadLoop() {
      setTimeout(gamePadLoop, 100);

      if (state.controlMode != "gamepad") {
        return;
      }

      var gamepads = navigator.getGamepads();

      for (var i = 0; i < gamepads.length; ++i) {
        var pad = gamepads[i];
        // Some slots are null until a button is pressed; timestamp==0 means
        // the slot has never reported input.
        if (!pad || pad.timestamp == 0) continue;

        var yaw   = applyDeadzone(pad.axes[0], 0.10);
        var alt   = applyDeadzone(-pad.axes[1], 0.10);
        var roll  = applyDeadzone(pad.axes[2], 0.10);
        var pitch = applyDeadzone(-pad.axes[3], 0.10);

        var aPressed = pad.buttons[0] && pad.buttons[0].pressed;
        var yPressed = pad.buttons[3] && pad.buttons[3].pressed;

        if (aPressed) {
          state.tele.user.angle = 0;
          state.tele.user.throttle = 0;
          state.tele.user.roll = 0;
          state.tele.user.altitude = 0;
          state.brakeOn = true;
          state.recording = false;
        } else {
          state.tele.user.angle = yaw;
          state.tele.user.throttle = limitedThrottle(pitch);
          state.tele.user.roll = Math.max(-1, Math.min(1, roll));
          state.tele.user.altitude = Math.max(-1, Math.min(1, alt));

          var anyInput = (yaw != 0 || pitch != 0 || roll != 0 || alt != 0);
          state.brakeOn = !anyInput;
          state.recording = anyInput;
        }

        // Y button cycles drive mode on rising edge (debounce against the
        // 100ms loop holding the button down across many ticks).
        if (yPressed && !prevYButton) {
          toggleDriveMode();
        }
        prevYButton = yPressed;

        postDrive();
        break;  // only consume the first connected pad
      }
    }


    // Send control updates to the server every .1 seconds.
    function joystickLoop () {
       setTimeout(function () {
            postDrive()

          if (joystickLoopRunning && state.controlMode == "joystick") {
             joystickLoop();
          }
       }, 100)
    }

    // Control throttle and steering with device orientation
    function handleOrientation(event) {

      var alpha = event.alpha;
      var beta = event.beta;
      var gamma = event.gamma;

      if (beta == null || gamma == null) {
        deviceHasOrientation = false;
        state.controlMode = "joystick";
        console.log("Invalid device orientation values, switched to joystick mode.")
      } else {
        deviceHasOrientation = true;
        console.log("device has valid orientation values")
      }

      updateUI();

      if(state.controlMode != "tilt" || !deviceHasOrientation || state.brakeOn){
        return;
      }

      if(!initialGamma && gamma) {
        initialGamma = gamma;
      }

      var newThrottle = gammaToThrottle(gamma);
      var newAngle = betaToSteering(beta, gamma);

      // prevent unexpected switch between full forward and full reverse
      // when device is parallel to ground
      if (state.tele.user.throttle > 0.9 && newThrottle <= 0) {
        newThrottle = 1.0
      }

      if (state.tele.user.throttle < -0.9 && newThrottle >= 0) {
        newThrottle = -1.0
      }

      state.tele.user.throttle = limitedThrottle(newThrottle);
      state.tele.user.angle = newAngle;
    }

    function deviceOrientationLoop () {
       setTimeout(function () {
          if(!state.brakeOn){
            postDrive()
          }

          if (state.controlMode == "tilt") {
            deviceOrientationLoop();
          }
       }, 100)
    }

    var throttleUp = function(){
      state.tele.user.throttle = limitedThrottle(Math.min(state.tele.user.throttle + .05, 1));
      postDrive()
    };

    var throttleDown = function(){
      state.tele.user.throttle = limitedThrottle(Math.max(state.tele.user.throttle - .05, -1));
      postDrive()
    };

    var throttleCenter = function(){
      state.tele.user.throttle = 0
      postDrive()
    };

    var angleLeft = function(){
      state.tele.user.angle = Math.max(state.tele.user.angle - .1, -1)
      postDrive()
    };

    var angleRight = function(){
      state.tele.user.angle = Math.min(state.tele.user.angle + .1, 1)
      postDrive()
    };

    var angleCenter = function(){
      state.tele.user.angle = 0
      postDrive()
    };

    var rollLeft = function(){
      state.tele.user.roll = Math.max(state.tele.user.roll - .1, -1)
      postDrive()
    };

    var rollRight = function(){
      state.tele.user.roll = Math.min(state.tele.user.roll + .1, 1)
      postDrive()
    };

    var rollCenter = function(){
      state.tele.user.roll = 0
      postDrive()
    };

    // Altitude is bipolar [-1, 1] where 0 = hover. Larger increments give
    // responsive stick feel; release (keyup) snaps back to 0.
    var altitudeUp = function(){
      state.tele.user.altitude = Math.min(state.tele.user.altitude + .1, 1)
      postDrive()
    };

    var altitudeDown = function(){
      state.tele.user.altitude = Math.max(state.tele.user.altitude - .1, -1)
      postDrive()
    };

    var altitudeCenter = function(){
      state.tele.user.altitude = 0
      postDrive()
    };

    var updateDriveMode = function(mode){
      state.driveMode = mode;
      postDrive(["drive_mode"])
    };

    var toggleDriveMode = function() {
      switch(state.driveMode) {
        case "user": {
            updateDriveMode("local_angle");
            break;
        }
        case "local_angle": {
            updateDriveMode("local");
            break;
        }
        default: {
            updateDriveMode("user");
            break;
        }
      }
    }

    var toggleRecording = function(){
      state.recording = !state.recording
      postDrive(['recording']);
    };

    var toggleBrake = function(){
      state.brakeOn = !state.brakeOn;
      initialGamma = null;

      if (state.brakeOn) {
        brake();
      }
    };

    var brake = function(i){
          console.log('post drive: ' + i)
          state.tele.user.angle = 0
          state.tele.user.throttle = 0
          state.tele.user.roll = 0
          state.tele.user.altitude = 0
          state.recording = false
          state.driveMode = 'user';
          postDrive()

      i++
      if (i < 5) {
        setTimeout(function () {
          console.log('calling brake:' + i)
          brake(i);
        }, 500)
      };

      state.brakeOn = true;
      updateUI();
    };

    var limitedThrottle = function(newThrottle){
      var limitedThrottle = 0;

      if (newThrottle > 0) {
        limitedThrottle = Math.min(state.maxThrottle, newThrottle);
      }

      if (newThrottle < 0) {
        limitedThrottle = Math.max((state.maxThrottle * -1), newThrottle);
      }

      if (state.throttleMode == 'constant') {
        limitedThrottle = state.maxThrottle;
      }

      return limitedThrottle;
    }


    // var drawLine = function(angle, throttle) {
    //
    //   throttleConstant = 100
    //   throttle = throttle * throttleConstant
    //   angleSign = Math.sign(angle)
    //   angle = toRadians(Math.abs(angle*90))
    //
    //   var canvas = document.getElementById("angleView"),
    //   context = canvas.getContext('2d');
    //   context.clearRect(0, 0, canvas.width, canvas.height);
    //
    //   base={'x':canvas.width/2, 'y':canvas.height}
    //
    //   pointX = Math.sin(angle) * throttle * angleSign
    //   pointY = Math.cos(angle) * throttle
    //   xPoint = {'x': pointX + base.x, 'y': base.y - pointY}
    //
    //   context.beginPath();
    //   context.moveTo(base.x, base.y);
    //   context.lineTo(xPoint.x, xPoint.y);
    //   context.lineWidth = 5;
    //   context.strokeStyle = '#ff0000';
    //   context.stroke();
    //   context.closePath();
    //
    // };

    var betaToSteering = function(beta, gamma) {
      const deadZone = 5;
      var angle = 0.0;
      var outsideDeadZone = false;
      var controlDirection = (Math.sign(initialGamma) * -1)

      //max steering angle at device 35º tilt
      var fullLeft = -35.0;
      var fullRight = 35.0;

      //handle beta 90 to 180 discontinuous transition at gamma 90
      if (beta > 90) {
        beta = (beta - 180) * Math.sign(gamma * -1) * controlDirection
      } else if (beta < -90) {
        beta = (beta + 180) * Math.sign(gamma * -1) * controlDirection
      }

      // set the deadzone for neutral sterring
      if (Math.abs(beta) > 90) {
        outsideDeadZone = Math.abs(beta) < 180 - deadZone;
      }
      else {
        outsideDeadZone = Math.abs(beta) > deadZone;
      }

      if (outsideDeadZone && beta < -90.0) {
        angle = remap(beta, fullLeft, (-180.0 + deadZone), -1.0, 0.0);
      }
      else if (outsideDeadZone && beta > 90.0) {
        angle = remap(beta, (180.0 - deadZone), fullRight, 0.0, 1.0);
      }
      else if (outsideDeadZone && beta < 0.0) {
        angle = remap(beta, fullLeft, 0.0 - deadZone, -1.0, 0);
      }
      else if (outsideDeadZone && beta > 0.0) {
        angle = remap(beta, 0.0 + deadZone, fullRight, 0.0, 1.0);
      }

      // set full turn if abs(angle) > 1
      if (angle < -1) {
        angle = -1;
      } else if (angle > 1) {
        angle = 1;
      }

      return angle * controlDirection;
    };

    var gammaToThrottle = function(gamma) {
      var throttle = 0.0;
      var gamma180 = gamma + 90;
      var initialGamma180 = initialGamma + 90;
      var controlDirection = (Math.sign(initialGamma) * -1);

      // 10 degree deadzone around the initial position
      // 45 degrees of motion for forward and reverse
      var minForward = Math.min((initialGamma180 + (5 * controlDirection)), (initialGamma180 + (50 * controlDirection)));
      var maxForward = Math.max((initialGamma180 + (5 * controlDirection)), (initialGamma180 + (50 * controlDirection)));
      var minReverse = Math.min((initialGamma180 - (50 * controlDirection)), (initialGamma180 - (5 * controlDirection)));
      var maxReverse = Math.max((initialGamma180 - (50 * controlDirection)), (initialGamma180 - (5 * controlDirection)));

      //constrain control input ranges to 0..180 continuous range
      minForward = Math.max(minForward, 0);
      maxForward = Math.min(maxForward, 180);
      minReverse = Math.max(minReverse, 0);
      maxReverse = Math.min(maxReverse, 180);

      if(gamma180 > minForward && gamma180 < maxForward) {
        // gamma in forward range
        if (controlDirection == -1) {
          throttle = remap(gamma180, minForward, maxForward, 1.0, 0.0);
        } else {
          throttle = remap(gamma180, minForward, maxForward, 0.0, 1.0);
        }
      } else if (gamma180 > minReverse && gamma180 < maxReverse) {
        // gamma in reverse range
        if (controlDirection == -1) {
          throttle = remap(gamma180, minReverse, maxReverse, 0.0, -1.0);
        } else  {
          throttle = remap(gamma180, minReverse, maxReverse, -1.0, 0.0);
        }
      }

      return throttle;
    };

}();


function toRadians (angle) {
  return angle * (Math.PI / 180);
}

function remap( x, oMin, oMax, nMin, nMax ){
  //range check
  if (oMin == oMax){
      console.log("Warning: Zero input range");
      return None;
  };

  if (nMin == nMax){
      console.log("Warning: Zero output range");
      return None
  }

  //check reversed input range
  var reverseInput = false;
  oldMin = Math.min( oMin, oMax );
  oldMax = Math.max( oMin, oMax );
  if (oldMin != oMin){
      reverseInput = true;
  }

  //check reversed output range
  var reverseOutput = false;
  newMin = Math.min( nMin, nMax )
  newMax = Math.max( nMin, nMax )
  if (newMin != nMin){
      reverseOutput = true;
  };

  var portion = (x-oldMin)*(newMax-newMin)/(oldMax-oldMin)
  if (reverseInput){
      portion = (oldMax-x)*(newMax-newMin)/(oldMax-oldMin);
  };

  var result = portion + newMin
  if (reverseOutput){
      result = newMax - portion;
  }

return result;
}
