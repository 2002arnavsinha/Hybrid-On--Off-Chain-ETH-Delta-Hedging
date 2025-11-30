// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// Import the official OpenZeppelin ERC20 contract
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

/**
 * @title HEDGE Token
 * @dev Basic ERC20 token used as collateral in our delta-hedging project.
 */
contract HedgeToken is ERC20 {
    constructor() ERC20("HEDGE Token", "HEDGE") {
        // Mint 1000 tokens (1000 * 10^18 units) to the deployer
        _mint(msg.sender, 1000 * 10 ** decimals());
    }
}