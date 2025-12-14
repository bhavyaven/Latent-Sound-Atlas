#include "oscmanager.h"
#include "soundpoint.h"
#include <iostream>
#include <cstring>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

// Helper to add 4-byte aligned string to buffer
static void addOSCString(std::vector<char>& buffer, const std::string& str) {
    buffer.insert(buffer.end(), str.begin(), str.end());
    buffer.push_back('\0');

    // Pad to 4-byte boundary
    while (buffer.size() % 4 != 0) {
        buffer.push_back('\0');
    }
}

// Helper to add float in big-endian format
static void addOSCFloat(std::vector<char>& buffer, float value) {
    // Copy float bytes into a byte array
    unsigned char bytes[4];
    memcpy(bytes, &value, 4);

    // Check if we're on little-endian system (most likely on x86/x64)
    uint32_t test = 1;
    bool isLittleEndian = (*((uint8_t*)&test) == 1);

    if (isLittleEndian) {
        // Reverse byte order for big-endian (network byte order)
        buffer.push_back(bytes[3]);
        buffer.push_back(bytes[2]);
        buffer.push_back(bytes[1]);
        buffer.push_back(bytes[0]);
    }
    else {
        // Already big-endian
        buffer.push_back(bytes[0]);
        buffer.push_back(bytes[1]);
        buffer.push_back(bytes[2]);
        buffer.push_back(bytes[3]);
    }
}

// Constructor
OSCManager::OSCManager(const std::string& host, int port)
    : host(host), port(port), connected(false), sockfd(-1)
{
    #ifdef _WIN32
        WSADATA wsaData;
        int result = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if (result != 0) {
            std::cerr << "WSAStartup failed: " << result << std::endl;
            return;
        }
    #endif

    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        std::cerr << "Failed to create socket" << std::endl;
        #ifdef _WIN32
            WSACleanup();
        #endif
        return;
    }

    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port);

    if (inet_pton(AF_INET, host.c_str(), &serverAddr.sin_addr) <= 0) {
        std::cerr << "Invalid address: " << host << std::endl;
        #ifdef _WIN32
            closesocket(sockfd);
            WSACleanup();
        #else
            close(sockfd);
        #endif
        sockfd = -1;
        return;
    }

    connected = true;
    std::cout << "OSC socket created for " << host << ":" << port << std::endl;
}

// Destructor
OSCManager::~OSCManager() {
    if (sockfd >= 0) {
    #ifdef _WIN32
        closesocket(sockfd);
        WSACleanup();
    #else
        close(sockfd);
    #endif
    }
}

bool OSCManager::isConnected() const {
    return connected && sockfd >= 0;
}

void OSCManager::playSoundByIndex(int index, const std::vector<SoundPoint>& points) {
    if (!connected || index < 0 || index >= static_cast<int>(points.size())) {
        std::cerr << "Invalid sound index or not connected" << std::endl;
        return;
    }

    const SoundPoint& p = points[index];

    std::cout << "Playing sound: " << p.synth
        << " freq=" << p.freq
        << " amp=" << p.amp << std::endl;

    playSound(p.synth, p.freq, p.amp);
}

void OSCManager::playSound(const std::string& synth, float freq, float amp) {
    if (!sendOSCMessage("/play_sound", synth, freq, amp)) {
        std::cerr << "Failed to send OSC message" << std::endl;
    }
}

bool OSCManager::sendOSCMessage(const std::string& address,
    const std::string& synth,
    float freq,
    float amp)
{
    if (!connected || sockfd < 0) {
        std::cerr << "OSC not connected" << std::endl;
        return false;
    }

    std::vector<char> buffer;

    // 1. OSC Address Pattern
    addOSCString(buffer, address);

    // 2. Type Tag String (comma + type chars)
    addOSCString(buffer, ",sff");  // string, float, float

    // 3. String Argument (synth name)
    addOSCString(buffer, synth);

    // 4. Float Arguments
    addOSCFloat(buffer, freq);
    addOSCFloat(buffer, amp);

    // Send via UDP
    int sent = sendto(sockfd,
        buffer.data(),
        buffer.size(),
        0,
        reinterpret_cast<struct sockaddr*>(&serverAddr),
        sizeof(serverAddr));

    if (sent < 0) {
        #ifdef _WIN32
            std::cerr << "Send failed with error: " << WSAGetLastError() << std::endl;
        #else
            std::cerr << "Send failed" << std::endl;
        #endif
        return false;
    }

    std::cout << "Sent OSC: " << address << " " << synth
        << " " << freq << " " << amp
        << " (" << sent << " bytes)" << std::endl;

    //// Debug outputfr
    //std::cout << "Message breakdown:" << std::endl;
    //std::cout << "  Address: " << address << std::endl;
    //std::cout << "  Type tags: ,sff" << std::endl;
    //std::cout << "  Arg1 (string): " << synth << std::endl;
    //std::cout << "  Arg2 (float): " << freq << std::endl;
    //std::cout << "  Arg3 (float): " << amp << std::endl;

    return sent == static_cast<int>(buffer.size());
}