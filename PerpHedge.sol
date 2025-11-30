// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";

/**
 *  @title PerpHedge
 *  @dev Delta-hedged perpetual futures for BTC options.
 *       Linear USD PnL, Chainlink oracle, packed storage, funding accrual.
 */
interface AggregatorV3Interface {
    function latestRoundData()
        external
        view
        returns (
            uint80 roundId,
            int256 answer,
            uint256 startedAt,
            uint256 updatedAt,
            uint80 answeredInRound
        );
}

interface IERC20 {
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function transfer(address to, uint256 amount) external returns (bool);
}

contract PerpHedge is Ownable {
    // --------------------------------------------------------------------- //
    //  STORAGE
    // --------------------------------------------------------------------- //
    AggregatorV3Interface internal immutable ethOracle;
    IERC20 public immutable collateralToken;

    struct PackedPosition {
        uint128 collateral;      // 128 bits (HEDGE wei)
        uint64  entryPrice;      // 64 bits  (BTC/USD * 1e8)
        uint32  leverage;        // 32 bits
        bool    isShort;         // 1 bit
        uint32  hedgeRatio;      // 32 bits (scaled 1e4 → 0.5 = 5000)
        uint64  lastFundingTime; // 64 bits (timestamp)
    }

    mapping(address => PackedPosition) private positions;

    int256 public fundingRate;          // 
    uint256 public sigmaOracle;         // scaled 1e4 (25% = 2500)

    // --------------------------------------------------------------------- //
    //  EVENTS
    // --------------------------------------------------------------------- //
    event PositionOpened(address indexed user, bool isShort, uint256 collateral, uint256 leverage, uint256 entryPrice);
    event PositionClosed(address indexed user, int256 pnl);
    event FundingApplied(int256 newFundingRate);
    event FundingAccrued(address indexed user, int256 funding);
    event AutoRebalanced(address indexed user, uint256 newEquity, uint32 newLeverage);
    event HedgeRatioUpdated(address indexed user, uint32 ratio);
    event VolatilityUpdated(uint256 newSigma);

    // --------------------------------------------------------------------- //
    //  CONSTRUCTOR
    // --------------------------------------------------------------------- //
    constructor(address _oracle, address _token) Ownable(msg.sender) {
        ethOracle = AggregatorV3Interface(_oracle);
        collateralToken = IERC20(_token);
    }

    // --------------------------------------------------------------------- //
    //  INTERNAL HELPERS
    // --------------------------------------------------------------------- //
    function _getETHPrice() internal view returns (uint256) {
        (, int256 price,,,) = ethOracle.latestRoundData();
        require(price > 0, "Invalid price");
        return uint256(price); // 8 decimals
    }

    // Apply funding to equity (accrues over time)
    function _applyFunding(PackedPosition storage p, address user) internal {
        if (p.lastFundingTime == 0) {
            p.lastFundingTime = uint64(block.timestamp);
            return;
        }

        uint256 elapsed = block.timestamp - p.lastFundingTime;
        if (elapsed < 8 hours) return;

        uint256 periods = elapsed / 8 hours;
        int256 size = int256(uint256(p.collateral)) * int256(uint256(p.leverage));
        int256 funding = size * fundingRate * int256(periods) / int256(1e18);

        if (funding < 0) {
            uint256 loss = uint256(-funding);
            p.collateral = loss > p.collateral ? 0 : p.collateral - uint128(loss);
        } else {
            p.collateral += uint128(uint256(funding));
        }

        p.lastFundingTime = uint64(block.timestamp);
        emit FundingAccrued(user, funding);
    }

    // --------------------------------------------------------------------- //
    //  PUBLIC VIEWERS
    // --------------------------------------------------------------------- //
    function ethOracleAddress() external view returns (address) {
        return address(ethOracle);
    }

    function getETHPricePublic() external view returns (uint256) {
        return _getETHPrice();
    }

    // --------------------------------------------------------------------- //
    //  OPEN POSITION
    // --------------------------------------------------------------------- //
    function openPosition(bool isShort, uint256 collateral, uint256 leverage) external {
        if (positions[msg.sender].collateral != 0) revert("Position exists");
        if (leverage < 1 || leverage > 10) revert("Invalid leverage");

        uint256 entryPrice = _getETHPrice();
        if (entryPrice > type(uint64).max) revert("Price overflow");

        if (!collateralToken.transferFrom(msg.sender, address(this), collateral))
            revert("Collateral transfer failed");

        positions[msg.sender] = PackedPosition({
            collateral: uint128(collateral),
            entryPrice: uint64(entryPrice),
            leverage: uint32(leverage),
            isShort: isShort,
            hedgeRatio: 0,
            lastFundingTime: uint64(block.timestamp)
        });

        emit PositionOpened(msg.sender, isShort, collateral, leverage, entryPrice);
    }

    // --------------------------------------------------------------------- //
    //  P&L (Linear USD PnL)
    // --------------------------------------------------------------------- //
    function getPnL(address user) public view returns (int256) {
        PackedPosition memory p = positions[user];
        if (p.collateral == 0) return 0;

        uint256 currentPrice = _getETHPrice();
        uint256 P0 = uint256(p.entryPrice);
        uint256 Pt = currentPrice;

        int256 size = int256(uint256(p.collateral)) * int256(uint256(p.leverage));

        if (p.isShort) {
            return size - (size * int256(Pt) / int256(P0));
        } else {
            return (size * int256(Pt) / int256(P0)) - size;
        }
    }

    // --------------------------------------------------------------------- //
    //  CLOSE POSITION
    // --------------------------------------------------------------------- //
    function closePosition() external {
        PackedPosition memory p = positions[msg.sender];
        require(p.collateral > 0, "No position");

        _applyFunding(positions[msg.sender], msg.sender); // Final accrual

        int256 pnl = getPnL(msg.sender);
        uint256 payout = p.collateral;

        if (pnl > 0) {
            payout += uint256(pnl);
        } else {
            uint256 loss = uint256(-pnl);
            payout = loss < payout ? payout - loss : 0;
        }

        delete positions[msg.sender];
        require(collateralToken.transfer(msg.sender, payout), "Transfer failed");
        emit PositionClosed(msg.sender, pnl);
    }

    // --------------------------------------------------------------------- //
    //  FUNDING RATE (vol-adjusted) - FIXED
    // --------------------------------------------------------------------- //
    function updateFundingRate(int256 baseRate) external onlyOwner {
        uint256 alpha = 5 * 10**17; // 0.5 in 1e18
        int256 sigmaScaled = int256(sigmaOracle) * 1e14; // 2800 → 2.8e17
        int256 volFactor = int256(1e18) + (int256(alpha) * sigmaScaled) / int256(1e18);
        fundingRate = baseRate * volFactor / int256(1e18);
        emit FundingApplied(fundingRate);
    }

    // --------------------------------------------------------------------- //
    //  VOLATILITY INPUT (off-chain)
    // --------------------------------------------------------------------- //
    function updateVolatility(uint256 newSigma) external onlyOwner {
        sigmaOracle = newSigma;
        emit VolatilityUpdated(newSigma);
    }

    // --------------------------------------------------------------------- //
    //  HEDGE RATIO INPUT (off-chain)
    // --------------------------------------------------------------------- //
    function updateHedgeRatio(uint32 ratio) external {
        PackedPosition storage p = positions[msg.sender];
        require(p.collateral > 0, "No position");
        p.hedgeRatio = ratio;
        emit HedgeRatioUpdated(msg.sender, ratio);
    }

    // --------------------------------------------------------------------- //
    //  AUTO-REBALANCE
    // --------------------------------------------------------------------- //
    function autoRebalance(address user) external {
        PackedPosition storage p = positions[user];
        require(p.collateral > 0, "No position");

        _applyFunding(p, user);

        int256 pnl = getPnL(user);
        uint256 equity;
        if (pnl >= 0) {
            equity = uint256(p.collateral) + uint256(pnl);
        } else {
            uint256 loss = uint256(-pnl);
            equity = loss > p.collateral ? 0 : uint256(p.collateral) - loss;
        }

        uint256 threshold = uint256(p.collateral) * 50 / 100;

        if (equity < threshold || p.hedgeRatio > 0) {
            uint32 newLev = p.hedgeRatio / 1000;
            if (newLev < 1) newLev = 1;
            if (newLev > 10) newLev = 10;

            p.leverage = newLev;
            p.collateral = uint128(equity);
            p.entryPrice = uint64(_getETHPrice());

            if (equity == 0) {
                delete positions[user];
            }

            emit AutoRebalanced(user, equity, newLev);
        }
    }

    // --------------------------------------------------------------------- //
    //  VIEW PACKED POSITION
    // --------------------------------------------------------------------- //
    function getPositionPacked(address user) external view returns (PackedPosition memory) {
        return positions[user];
    }
}