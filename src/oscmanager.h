#pragma once
#include <string>
#include <vector>
#include "SoundPoint.h"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <netinet/in.h>
#include <arpa/inet.h>
#endif

class OSCManager {
public:
    OSCManager(const std::string& host, int port);
    ~OSCManager();

    bool isConnected() const;
    void playSoundByIndex(int index, const std::vector<SoundPoint>& points);
    void playSound(const std::string& synth, float freq, float amp);

private:
    bool sendOSCMessage(const std::string& address,
        const std::string& synth,
        float freq, float amp);
int sockfd;
    bool connected;

    struct sockaddr_in serverAddr;

    std::string host;
    int port;
};