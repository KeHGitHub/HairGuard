import AppKit
import Foundation


final class HairGuardAlertWindow: NSWindow {
    var dismissAction: (() -> Void)?

    override var canBecomeKey: Bool { true }

    override func keyDown(with event: NSEvent) {
        dismissAction?()
    }
}


final class HairGuardStatusController: NSObject, NSApplicationDelegate, @unchecked Sendable {
    private var statusItem: NSStatusItem!
    private var alertWindow: HairGuardAlertWindow?
    private let pulseItem = NSMenuItem(title: "Pulse avg: acquiring...", action: nil, keyEquivalent: "")
    private let detectorItem = NSMenuItem(
        title: "Scratch detector: IDLE",
        action: nil,
        keyEquivalent: ""
    )

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)

        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        if let button = statusItem.button {
            button.image = NSImage(
                systemSymbolName: "waveform.path.ecg",
                accessibilityDescription: "HairGuard status"
            )
            button.image?.isTemplate = true
            button.toolTip = "HairGuard"
        }

        pulseItem.isEnabled = false
        detectorItem.isEnabled = false

        let menu = NSMenu()
        menu.addItem(pulseItem)
        menu.addItem(detectorItem)
        menu.addItem(.separator())

        let quitItem = NSMenuItem(
            title: "Quit HairGuard",
            action: #selector(quitHairGuard),
            keyEquivalent: "q"
        )
        quitItem.keyEquivalentModifierMask = [.command]
        quitItem.target = self
        menu.addItem(quitItem)
        statusItem.menu = menu

        DispatchQueue.global(qos: .utility).async { [weak self] in
            self?.readStatusUpdates()
        }
    }

    private func readStatusUpdates() {
        while let line = readLine() {
            let parts = line.split(separator: "\t", maxSplits: 1, omittingEmptySubsequences: false)
            guard parts.count == 2 else { continue }
            let name = String(parts[0])
            let value = String(parts[1])
            DispatchQueue.main.async { [weak self] in
                self?.apply(name: name, value: value)
            }
        }
        DispatchQueue.main.async {
            NSApp.terminate(nil)
        }
    }

    private func apply(name: String, value: String) {
        switch name {
        case "PULSE":
            pulseItem.title = "Pulse avg: \(value.lowercased())"
        case "STATE":
            detectorItem.title = "Scratch detector: \(value)"
        case "ALERT":
            if value == "SHOW" {
                showAlert()
            } else if value == "HIDE" {
                hideAlert(notifyParent: false)
            }
        default:
            break
        }
    }

    private func showAlert() {
        if let window = alertWindow {
            NSApp.activate(ignoringOtherApps: true)
            window.makeKeyAndOrderFront(nil)
            window.orderFrontRegardless()
            return
        }
        guard let screen = NSScreen.main ?? NSScreen.screens.first else { return }

        let window = HairGuardAlertWindow(
            contentRect: screen.frame,
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        window.level = .screenSaver
        window.backgroundColor = NSColor(
            calibratedRed: 215.0 / 255.0,
            green: 25.0 / 255.0,
            blue: 63.0 / 255.0,
            alpha: 1.0
        )
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        window.isOpaque = true
        window.hidesOnDeactivate = false
        window.dismissAction = { [weak self] in
            self?.hideAlert(notifyParent: true)
        }

        let content = NSView(frame: screen.frame)
        let stopLabel = NSTextField(labelWithString: "STOP")
        stopLabel.font = NSFont.systemFont(ofSize: 104, weight: .bold)
        stopLabel.textColor = .white
        stopLabel.alignment = .center
        stopLabel.translatesAutoresizingMaskIntoConstraints = false

        let instructionLabel = NSTextField(
            labelWithString: "Move your hand away from your head  •  Any key dismisses"
        )
        instructionLabel.font = NSFont.systemFont(ofSize: 18, weight: .regular)
        instructionLabel.textColor = .white
        instructionLabel.alignment = .center
        instructionLabel.translatesAutoresizingMaskIntoConstraints = false

        content.addSubview(stopLabel)
        content.addSubview(instructionLabel)
        NSLayoutConstraint.activate([
            stopLabel.centerXAnchor.constraint(equalTo: content.centerXAnchor),
            stopLabel.centerYAnchor.constraint(equalTo: content.centerYAnchor, constant: -25),
            instructionLabel.centerXAnchor.constraint(equalTo: content.centerXAnchor),
            instructionLabel.topAnchor.constraint(equalTo: stopLabel.bottomAnchor, constant: 45),
        ])
        window.contentView = content
        alertWindow = window

        NSApp.activate(ignoringOtherApps: true)
        window.makeKeyAndOrderFront(nil)
        window.orderFrontRegardless()
    }

    private func hideAlert(notifyParent: Bool) {
        alertWindow?.orderOut(nil)
        if notifyParent {
            sendEvent("DISMISSED")
        }
    }

    private func sendEvent(_ event: String) {
        FileHandle.standardOutput.write(Data("\(event)\n".utf8))
        try? FileHandle.standardOutput.synchronize()
    }

    @objc private func quitHairGuard() {
        sendEvent("QUIT")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            NSApp.terminate(nil)
        }
    }
}


let application = NSApplication.shared
let controller = HairGuardStatusController()
application.delegate = controller
application.run()
