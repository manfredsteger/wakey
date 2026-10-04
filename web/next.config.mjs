/** @type {import('next').NextConfig} */
const config = {
  serverExternalPackages: ['@prisma/client'],
  agentRules: false,
  // Allow dev-server access from local network devices (iPhone, tablet, etc.)
  allowedDevOrigins: [
    '192.168.0.0/16',
    '10.0.0.0/8',
    '172.16.0.0/12',
  ],
};

export default config;
