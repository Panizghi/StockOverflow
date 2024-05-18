// This script interacts directly with a NEAR smart contract to get and set greetings.

async function initNear() {
    // Configuration for the NEAR connection
    const nearConfig = {
        networkId: "testnet",
        keyStore: new nearAPI.keyStores.BrowserLocalStorageKeyStore(),
        nodeUrl: "https://rpc.testnet.near.org",
        walletUrl: "https://wallet.testnet.near.org",
        helperUrl: "https://helper.testnet.near.org",
        explorerUrl: "https://explorer.testnet.near.org",
    };

    // Initialize a connection to the NEAR blockchain
    const near = await nearAPI.connect(nearConfig);
    const walletConnection = new nearAPI.WalletConnection(near);

    // Assuming the contract is deployed with the following ID
    const contract = new nearAPI.Contract(walletConnection.account(), "hello.near-examples.near", {
        // View methods read from the blockchain without making changes.
        viewMethods: ['get_greeting'],
        // Change methods modify or add data to the blockchain.
        changeMethods: ['set_greeting'],
        // Sender is the account that signed in to the wallet
        sender: walletConnection.getAccountId(),
    });

    return { walletConnection, contract };
}

async function fetchGreeting() {
    try {
        const { contract } = await initNear();
        const greeting = await contract.get_greeting({});
        document.getElementById('greeting').textContent = greeting;
    } catch (error) {
        console.error("Failed to fetch greeting:", error);
    }
}

async function updateGreeting() {
    const newGreeting = document.getElementById('newGreeting').value;

    try {
        const { walletConnection, contract } = await initNear();

        if (walletConnection.isSignedIn()) {
            await contract.set_greeting({ greeting: newGreeting });
            fetchGreeting(); // Refresh the greeting displayed
            document.getElementById('newGreeting').value = ''; // Clear the input field
        } else {
            // If the user isn't signed in, redirect them to the wallet to sign in.
            walletConnection.requestSignIn({
                contractId: "hello.near-examples.near",
                methodNames: ['set_greeting'] // Specify which methods this app can call
            });
        }
    } catch (error) {
        console.error("Failed to update greeting:", error);
    }
}

document.addEventListener('DOMContentLoaded', function() {
    // Fetch and display the current greeting when the page loads.
    fetchGreeting();
});
