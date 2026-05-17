// XboxBridge - reads Xbox controller via Apple's GameController framework
// and forwards state to a Unix datagram socket at 60Hz.
//
// Required because Apple's `com.apple.gamecontroller.driver.XboxGamepad`
// dext claims exclusive ownership of Xbox controllers on modern macOS,
// blocking pygame/SDL/hidapi paths. GameController.framework is the only
// supported way in, and it requires a real .app bundle.
//
// Frame format (26 bytes, little-endian):
//   float32 leftX, float32 leftY, float32 rightX, float32 rightY,
//   float32 leftTrigger, float32 rightTrigger,
//   uint8 buttons (bit0=A, bit1=B), uint8 connected (1/0)
//
// LT (leftTrigger) is a reset button: on rising edge past 0.5, the drone
// teleports back to its spawn position.
//
// Socket path: $NANOPILOT_XBOX_SOCK or /tmp/nanopilot_xbox.sock
//
// Logs go to NSLog (visible via `log stream --process XboxBridge`).
// FileHandle.standardError writes are dropped for .app launched via `open`.

import AppKit
import GameController
import Darwin
import os

let SOCK_PATH = ProcessInfo.processInfo.environment["NANOPILOT_XBOX_SOCK"]
    ?? "/tmp/nanopilot_xbox.sock"

let osLogger = OSLog(subsystem: "com.nanopilot.xboxbridge", category: "bridge")

func logLine(_ s: String) {
    // Use public format so the message isn't redacted in `log stream`.
    os_log("[XboxBridge] %{public}@", log: osLogger, type: .default, s)
}

final class Bridge: NSObject {
    var controllers: [GCController] = []
    var sockFD: Int32 = -1
    var sockAddr = sockaddr_un()
    var addrLen: socklen_t = 0
    var timer: Timer?
    var statusTimer: Timer?

    // Latest values (updated from valueChangedHandler OR polled fallback).
    var lX: Float = 0
    var lY: Float = 0
    var rX: Float = 0
    var rY: Float = 0
    var lT: Float = 0
    var rT: Float = 0
    var buttons: UInt8 = 0
    var lastChangeAt: Date = .distantPast
    var changeCount: Int = 0

    override init() {
        super.init()
        if !setupSocket() {
            logLine("socket setup failed; exiting")
            exit(1)
        }

        // macOS 11.3+: required for .accessory / LSUIElement apps to receive
        // controller events without being the focused frontmost app.
        if #available(macOS 11.3, *) {
            GCController.shouldMonitorBackgroundEvents = true
            logLine("shouldMonitorBackgroundEvents = true")
        }

        NotificationCenter.default.addObserver(
            self, selector: #selector(controllerConnected),
            name: .GCControllerDidConnect, object: nil)
        NotificationCenter.default.addObserver(
            self, selector: #selector(controllerDisconnected),
            name: .GCControllerDidDisconnect, object: nil)

        // Wired controllers are usually already enumerated at launch. Attach
        // to ALL of them — multiple "logical" devices may be exposed.
        let initial = GCController.controllers()
        logLine("initial controllers count = \(initial.count)")
        for c in initial { attach(c) }

        // Tell the framework to start watching for accessories that aren't
        // already enumerated (e.g. wireless controllers paired but not yet
        // delivering the connect notification).
        GCController.startWirelessControllerDiscovery {
            logLine("startWirelessControllerDiscovery completion")
        }

        timer = Timer.scheduledTimer(withTimeInterval: 1.0/60.0, repeats: true) {
            [weak self] _ in
            self?.tick()
        }
        statusTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) {
            [weak self] _ in
            self?.logStatus()
        }
        logLine("ready, sending to \(SOCK_PATH)")
    }

    func setupSocket() -> Bool {
        sockFD = socket(AF_UNIX, SOCK_DGRAM, 0)
        if sockFD < 0 { return false }
        sockAddr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = Array(SOCK_PATH.utf8)
        if pathBytes.count >= MemoryLayout.size(ofValue: sockAddr.sun_path) {
            logLine("socket path too long")
            return false
        }
        withUnsafeMutableBytes(of: &sockAddr.sun_path) { rawPtr in
            for (i, b) in pathBytes.enumerated() {
                rawPtr[i] = b
            }
            rawPtr[pathBytes.count] = 0
        }
        addrLen = socklen_t(MemoryLayout<sockaddr_un>.size)
        return true
    }

    @objc func controllerConnected(_ note: Notification) {
        if let c = note.object as? GCController {
            attach(c)
        }
    }
    @objc func controllerDisconnected(_ note: Notification) {
        if let c = note.object as? GCController {
            controllers.removeAll(where: { $0 === c })
            logLine("controller disconnected: \(c.vendorName ?? "<unknown>")")
        }
    }

    func attach(_ c: GCController) {
        if controllers.contains(where: { $0 === c }) { return }
        controllers.append(c)
        let name = c.vendorName ?? "<unknown>"
        let category: String
        if #available(macOS 11.0, *) {
            category = c.productCategory
        } else {
            category = "?"
        }
        let kind = (c.extendedGamepad != nil) ? "extendedGamepad" : "other"
        logLine("controller connected: name='\(name)' category='\(category)' kind=\(kind)")

        // Path 1 — legacy GCExtendedGamepad valueChangedHandler.
        if let g = c.extendedGamepad {
            g.valueChangedHandler = { [weak self] (gp, elem) in
                guard let self = self else { return }
                self.lX = gp.leftThumbstick.xAxis.value
                self.lY = gp.leftThumbstick.yAxis.value
                self.rX = gp.rightThumbstick.xAxis.value
                self.rY = gp.rightThumbstick.yAxis.value
                self.lT = gp.leftTrigger.value
                self.rT = gp.rightTrigger.value
                var b: UInt8 = 0
                if gp.buttonA.isPressed { b |= 0x01 }
                if gp.buttonB.isPressed { b |= 0x02 }
                self.buttons = b
                self.lastChangeAt = Date()
                self.changeCount += 1
            }
            logLine("extendedGamepad valueChangedHandler installed")
        }

        // Path 2 — GameController 2.0+ unified physicalInputProfile API.
        // Different code path; if the legacy callback never fires, this
        // sometimes does (and lets us discover what element names the
        // dext is actually exposing).
        if #available(macOS 11.0, *) {
            let profile = c.physicalInputProfile
            logLine("physicalInputProfile elements: \(profile.elements.keys.sorted())")
            profile.valueDidChangeHandler = { [weak self] (_, elem) in
                guard let self = self else { return }
                let name = elem.localizedName ?? elem.aliases.first ?? "?"
                self.lastChangeAt = Date()
                self.changeCount += 1
                if self.changeCount <= 30 || self.changeCount % 60 == 0 {
                    if let axis = elem as? GCControllerAxisInput {
                        logLine("profile change: \(name) axis=\(axis.value)")
                    } else if let btn = elem as? GCControllerButtonInput {
                        logLine("profile change: \(name) button=\(btn.value) pressed=\(btn.isPressed)")
                    } else {
                        logLine("profile change: \(name) (\(type(of: elem)))")
                    }
                }
            }
            logLine("physicalInputProfile.valueDidChangeHandler installed")
        }

        // Path 3 — controller pause handler (deprecated but sometimes the only
        // thing that fires on certain dext-managed devices).
        c.controllerPausedHandler = { _ in
            logLine("controllerPausedHandler fired")
        }
    }

    func tick() {
        // Keep latest values from polling too — if event-driven path is
        // silent the polling path will at least surface what the framework
        // currently believes the values are.
        if let g = controllers.first?.extendedGamepad {
            // Note: this read may show 0 even when sticks are moved if the
            // dext only updates state at event delivery time.
            let pollLY = g.leftThumbstick.yAxis.value
            let pollLX = g.leftThumbstick.xAxis.value
            let pollRX = g.rightThumbstick.xAxis.value
            let pollRY = g.rightThumbstick.yAxis.value
            let pollLT = g.leftTrigger.value
            let pollRT = g.rightTrigger.value
            // Don't overwrite event-driven values with stale polled zeros —
            // only adopt polled values if they're non-zero or we've never
            // received an event-driven update.
            if abs(pollLX) + abs(pollLY) + abs(pollRX) + abs(pollRY) + pollRT + pollLT > 0.0001
                || lastChangeAt == .distantPast {
                lX = pollLX; lY = pollLY; rX = pollRX; rY = pollRY; lT = pollLT; rT = pollRT
                var b: UInt8 = 0
                if g.buttonA.isPressed { b |= 0x01 }
                if g.buttonB.isPressed { b |= 0x02 }
                buttons = b
            }
        }

        let connected: UInt8 = controllers.isEmpty ? 0 : 1
        var buf = Data(capacity: 26)
        var f = lX; buf.append(Data(bytes: &f, count: 4))
        f = lY;     buf.append(Data(bytes: &f, count: 4))
        f = rX;     buf.append(Data(bytes: &f, count: 4))
        f = rY;     buf.append(Data(bytes: &f, count: 4))
        f = lT;     buf.append(Data(bytes: &f, count: 4))
        f = rT;     buf.append(Data(bytes: &f, count: 4))
        buf.append(buttons)
        buf.append(connected)

        let sent: Int = buf.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> Int in
            withUnsafePointer(to: &sockAddr) { (addrPtr) -> Int in
                let saPtr = UnsafeRawPointer(addrPtr)
                    .assumingMemoryBound(to: sockaddr.self)
                return sendto(sockFD, raw.baseAddress, buf.count, 0, saPtr, addrLen)
            }
        }
        _ = sent
    }

    func logStatus() {
        let polled: String
        if let g = controllers.first?.extendedGamepad {
            polled = String(format: "poll[lX=%+.2f lY=%+.2f rX=%+.2f rY=%+.2f lT=%.2f rT=%.2f A=%d B=%d]",
                            g.leftThumbstick.xAxis.value,
                            g.leftThumbstick.yAxis.value,
                            g.rightThumbstick.xAxis.value,
                            g.rightThumbstick.yAxis.value,
                            g.leftTrigger.value,
                            g.rightTrigger.value,
                            g.buttonA.isPressed ? 1 : 0,
                            g.buttonB.isPressed ? 1 : 0)
        } else {
            polled = "poll[no extendedGamepad]"
        }
        let evtAge = lastChangeAt == .distantPast
            ? "never"
            : String(format: "%.1fs", Date().timeIntervalSince(lastChangeAt))
        logLine("status controllers=\(controllers.count) changeCount=\(changeCount) lastChange=\(evtAge) \(polled)")
    }
}

class AppDelegate: NSObject, NSApplicationDelegate {
    var bridge: Bridge?
    func applicationDidFinishLaunching(_ note: Notification) {
        bridge = Bridge()
        // Activate the app so input events are delivered. Even with
        // .accessory policy, activate is needed on some macOS versions
        // for the GameController dext to start emitting events.
        NSApp.activate(ignoringOtherApps: true)
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.accessory)
let delegate = AppDelegate()
app.delegate = delegate
app.run()
