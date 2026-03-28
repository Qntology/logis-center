// Access global libraries loaded via index.html
const ethers = (window as any).ethers;
const pako = (window as any).pako;

export function crc32(s: string, polynomial: number = 0x04C11DB7, initialValue: number = 0xFFFFFFFF, finalXORValue: number = 0xFFFFFFFF): number {
    let crc = initialValue;
    const table: number[] = [];
    let i, j, c;

    function reverse(x: number, n: number): number {
        let b = 0;
        while (n) {
            b = b * 2 + x % 2;
            x /= 2;
            x -= x % 1;
            n--;
        }
        return b;
    }

    for (i = 256; i >= 0; i--) {
        c = reverse(i, 32);
        for (j = 0; j < 8; j++) {
            c = ((c * 2) ^ (((c >>> 31) % 2) * polynomial)) >>> 0;
        }
        table[i] = reverse(c, 32);
    }

    for (i = 0; i < s.length; i++) {
        c = s.charCodeAt(i);
        if (c > 255) {
            throw new RangeError();
        }
        j = (crc % 256) ^ c;
        crc = ((crc / 256) ^ table[j]) >>> 0;
    }
    return (crc ^ finalXORValue) >>> 0;
}

export function hashId(text: string): string {
    if (typeof text === "undefined" || text === null) {
        const account = ethers.Wallet.createRandom();
        text = account.privateKey;
    }
    const hashMessage = ethers.hashMessage(text);
    return ethers.computeAddress(hashMessage).toLowerCase();
}

export function safeClone<T>(obj: T): T | null {
    const seen = new WeakMap();
    function clone(value: any): any {
        if (typeof value !== "object" || value === null) return value;
        if (seen.has(value)) return null; // Remove circular references
        const copy: any = Array.isArray(value) ? [] : {};
        seen.set(value, copy);
        for (const key in value) {
            copy[key] = clone(value[key]);
        }
        return copy;
    }
    return clone(obj);
}

export function time2text(date: number | string | Date): string {
    const d = new Date(date);
    const now = new Date();
    const seconds = Math.floor((now.getTime() - d.getTime()) / 1000);

    const intervalYear = seconds / 31536000;
    if (intervalYear > 1) return Math.floor(intervalYear) + " years";

    const intervalMonth = seconds / 2592000;
    if (intervalMonth > 1) return Math.floor(intervalMonth) + " months";

    const intervalDay = seconds / 86400;
    if (intervalDay > 1) return Math.floor(intervalDay) + " days";

    const intervalHour = seconds / 3600;
    if (intervalHour > 1) return Math.floor(intervalHour) + " hours";

    const intervalMinute = seconds / 60;
    if (intervalMinute > 1) return Math.floor(intervalMinute) + " minutes";

    return Math.floor(seconds) + " seconds";
}

export function parseStatus(status: number | string): string {
    // Check if it's already a string representation
    if (typeof status === 'string') {
        // If it's already a string like "progress", return it
        if (isNaN(Number(status))) return status;
        // If it's "1", convert to number and proceed
        status = Number(status);
    }

    let step = '';
    switch (status) {
        case 1: step = 'progress'; break;
        case 2: step = 'stop'; break;
        case 3: step = 'cancel'; break;
        case 4: step = 'refund'; break;
        case 5: step = 'return'; break;
        case 6: step = 'error'; break;
        case 7: step = 'expire'; break;
        case 8: step = 'exchange'; break;
        case 9: step = 'complete'; break;
        case 10: step = 'draft'; break;
        case 11: step = 'show'; break;
        case 12: step = 'hide'; break;
        default: step = ''; // fallback
    }
    return step;
}

export function randomHash(msg: string = Math.random() + ""): string {
    return crc32(msg).toString(32);
}

// --- Data Adapter for Rust/LanceDB ---

/**
 * Parsed item data handling both raw JSON objects and pako-compressed binary/base64.
 */
export function parseItemData(rawData: any): any {
    if (!rawData) return {};

    // 1. Already an object (Parsed by Rust serde)
    if (typeof rawData === 'object' && !rawData.buffer && !Array.isArray(rawData)) {
        return rawData;
    }

    // 2. JSON String
    if (typeof rawData === 'string') {
        try {
            // Attempt standard JSON parse first
            return JSON.parse(rawData);
        } catch (e) {
            // Not a JSON string, might be Base64 compressed string?
            // In the Rust code `upsert_item`, we see data can be stored as JSON string.
            // If pako was used, it might be base64 encoded.
            try {
                // If window.atob fails, it's not base64
                // const binaryString = window.atob(rawData);
                // const bytes = new Uint8Array(binaryString.length);
                // for (let i = 0; i < binaryString.length; i++) {
                //     bytes[i] = binaryString.charCodeAt(i);
                // }
                // const decompressed = pako.ungzip(bytes, { to: 'string' });
                // return JSON.parse(decompressed);
                
                // For now, return as is if string parse fails, or handle specific base64 cases if detected
                return { text: rawData }; 
            } catch (err) {
                return {};
            }
        }
    }

    // 3. Uint8Array / ArrayBuffer (Raw Pako data)
    if (rawData instanceof Uint8Array || rawData instanceof ArrayBuffer || (Array.isArray(rawData) && typeof rawData[0] === 'number')) {
        try {
            const decompressed = pako.ungzip(rawData, { to: 'string' });
            return JSON.parse(decompressed);
        } catch (err) {
            console.warn("Failed to decompress binary data:", err);
            return {};
        }
    }

    return {};
}
